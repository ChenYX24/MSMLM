# tox21_eval_csv_fewshot_detailed.py
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from MSMLM.code.modules.mol_aware_lm import MolAwareCausalLM
import torch.nn.functional as F
from tqdm import tqdm
import random

# ---------------- Config ----------------
MODEL_PATH = "/data1/chenyuxuan/Project/MSMLM/code/mol_sft/outputs/sft"
CSV_PATH   = "/data1/lvchangwei/GVP_finetune/MoleculeNet/Classfication/Tox21/tox21.csv"
MOL_TOKEN  = "<mol>"
DEVICE     = "cuda"
BATCH_SIZE = 16
FEWSHOT_K  = 3  # few-shot 示例数量

# ---------------- Utils ----------------
def _tie_if_needed(llm):
    try: llm.tie_weights()
    except Exception: pass
    try:
        llm.get_output_embeddings().weight = llm.get_input_embeddings().weight
    except Exception: pass

def _check_tying(llm):
    try:
        w_in  = llm.get_input_embeddings().weight
        w_out = llm.get_output_embeddings().weight
        return w_in.data_ptr() == w_out.data_ptr()
    except Exception:
        return True

def load_model(model_path: str, mol_token: str = "<mol>"):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
    ).eval()

    old_vocab = llm.get_input_embeddings().num_embeddings
    new_vocab = len(tok)
    if new_vocab != old_vocab:
        print(f"[info] resize_token_embeddings: {old_vocab} -> {new_vocab}")
        llm.resize_token_embeddings(new_vocab)
        _tie_if_needed(llm)

    for t in ["eos_token_id", "pad_token_id", "bos_token_id"]:
        if getattr(llm.config, t) is None and getattr(tok, t.replace("_id","")) is not None:
            setattr(llm.config, t, getattr(tok, t.replace("_id","")))

    if not _check_tying(llm):
        print("[warn] embeddings and lm_head are not tied. Retie now.")
        _tie_if_needed(llm)

    model = MolAwareCausalLM(llm=llm, tokenizer=tok, mol_token=mol_token, debug=False)
    return tok, model

# ---------------- Verbalizer & scoring ----------------
def _best_verbalizers(tok):
    candidates = [
        (" Yes", " No"), (" yes", " no"),
        (" 1", " 0"), (" True", " False")
    ]
    for y, n in candidates:
        y_ids = tok(y, add_special_tokens=False)["input_ids"]
        n_ids = tok(n, add_special_tokens=False)["input_ids"]
        if len(y_ids) == 1 and len(n_ids) == 1:
            return y_ids, n_ids
    return tok(candidates[0][0], add_special_tokens=False)["input_ids"], tok(candidates[0][1], add_special_tokens=False)["input_ids"]

@torch.no_grad()
def _score_label_seq(llm, input_ids, label_ids):
    device = input_ids.device
    full = torch.cat([input_ids, torch.tensor([label_ids], device=device)], dim=1)
    out = llm(full[:, :-1])
    logits = out.logits[:, -len(label_ids):, :]
    logprobs = F.log_softmax(logits, dim=-1)
    gather = logprobs[0, torch.arange(len(label_ids)), torch.tensor(label_ids, device=device)]
    return float(gather.sum().item())

# ---------------- Few-shot prompt construction with detailed explanation ----------------
def construct_fewshot_prompt(task_name, smiles, fewshot_examples, mol_token=MOL_TOKEN):
    explanation = (
        "This task is part of the Tox21 dataset, which measures the toxicity of molecules on human cell lines. "
        "A molecule is considered positive if it shows significant activation in bioassays, and negative otherwise. "
        "Labels are binary: 'Yes' for positive/toxic, 'No' for negative/non-toxic."
    )
    prompt = f"Task: {task_name} (Tox21)\nExplanation: {explanation}\nExamples:\n"
    for ex_smi, ex_label in fewshot_examples:
        prompt += f"Molecule: {ex_smi}{mol_token}\nLabel: {ex_label}\n"
    prompt += f"Now predict for the following molecule:\nMolecule: {smiles}{mol_token}\nLabel: "
    return prompt

# ---------------- Batched inference ----------------
@torch.no_grad()
def predict_logits_batch_fewshot(tok, model, smiles_list, labels_list, task_name=None, batch_size=16, k=FEWSHOT_K):
    dev = next(model.parameters()).device
    y_ids, n_ids = _best_verbalizers(tok)
    probs = []

    # 准备 few-shot pool
    pool = [(smi, str(lbl)) for smi, lbl in zip(smiles_list, labels_list) if lbl in [0,1]]

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Predicting", ncols=100):
        batch = smiles_list[i:i+batch_size]
        batch_prompts = []

        for smi in batch:
            fewshot_examples = random.sample(pool, min(k, len(pool)))
            prompt = construct_fewshot_prompt(
                task_name=task_name if task_name else "Binary classification",
                smiles=smi,
                fewshot_examples=fewshot_examples,
                mol_token=MOL_TOKEN
            )
            batch_prompts.append(prompt)

        # tokenizer
        inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(dev)
        batch_probs = []

        for b_idx in range(len(batch)):
            input_ids = inputs["input_ids"][b_idx:b_idx+1]

            if len(y_ids) == 1 and len(n_ids) == 1:
                out = model.llm(**{"input_ids": input_ids})
                logits = out.logits[0, -1, :]
                two = torch.stack([logits[y_ids[0]], logits[n_ids[0]]], dim=0)
                p_yes = torch.softmax(two, dim=0)[0].item()
            else:
                ly = _score_label_seq(model.llm, input_ids, y_ids)
                ln = _score_label_seq(model.llm, input_ids, n_ids)
                m = max(ly, ln)
                p_yes = np.exp(ly - m) / (np.exp(ly - m) + np.exp(ln - m))

            batch_probs.append(p_yes)
        probs.extend(batch_probs)

    return np.array(probs, dtype=np.float32)

# ---------------- Evaluation ----------------
def evaluate_tox21_csv_fewshot(tok, model, csv_path: str, batch_size=16, k=FEWSHOT_K):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["smiles"])
    df = df[df["smiles"].str.strip() != ""]
    df = df.drop_duplicates(subset=["smiles"])
    print(f"[data] total samples after cleaning = {len(df)}")

    tasks = [c for c in df.columns if c != "smiles"]
    results = {}

    for task in tasks:
        print(f"\n[eval] Task = {task}")
        smiles = df["smiles"].values
        labels = df[task].values
        mask = (labels == 0) | (labels == 1)
        smiles = [s for s, m in zip(smiles, mask) if m]
        task_labels = labels[mask]
        print(f"  valid samples = {len(task_labels)}")
        if len(task_labels) == 0:
            print(f"  [skip] no valid labels for {task}")
            continue

        probs = predict_logits_batch_fewshot(tok, model, smiles, task_labels, task_name=task, batch_size=batch_size, k=k)
        auc = roc_auc_score(task_labels, probs)
        results[task] = auc
        print(f"  AUC = {auc:.4f}")

    print("\n=== Summary ===")
    if results:
        mean_auc = np.mean(list(results.values()))
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        print(f"Mean AUC = {mean_auc:.4f}")
    else:
        print("No valid tasks to evaluate.")
    return results

# ---------------- Main ----------------
if __name__ == "__main__":
    tok, model = load_model(MODEL_PATH, mol_token=MOL_TOKEN)
    evaluate_tox21_csv_fewshot(tok, model, CSV_PATH, batch_size=BATCH_SIZE, k=FEWSHOT_K)
