# sft_tester.py
# -*- coding: utf-8 -*-
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.mol_aware_lm_simple import MolAwareCausalLM

os.chdir(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "/data1/chenyuxuan/Project/MSMLM/code/mol_sft/outputs/sft"
MOL_TOKEN  = "<mol>"

def _tie_if_needed(llm):
    """确保 lm_head 与 embed_tokens 权重 tying，防止复读退化。"""
    try:
        llm.tie_weights()
    except Exception:
        pass
    # 强制共享（有些模型类的 tie_weights 不生效）
    try:
        llm.get_output_embeddings().weight = llm.get_input_embeddings().weight
    except Exception:
        pass

def _check_tying(llm):
    """返回是否真正 tied（通过 data_ptr 判断共享内存）。"""
    try:
        w_in  = llm.get_input_embeddings().weight
        w_out = llm.get_output_embeddings().weight
        return w_in.data_ptr() == w_out.data_ptr()
    except Exception:
        return True  # 某些实现取不到，先不阻断

def load_model(model_path: str, mol_token: str = "<mol>"):
    # 1) 和你第一段保持一致的精度/设备；避免 device_map="auto" 引入的额外不确定性
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,          # 和第一段一致
        device_map={"": "cuda:0"},          # 固定到 cuda:0
    ).eval()

    # 2) vocab 对齐：仅在不一致时才 resize，并立刻 re-tie
    old_vocab = llm.get_input_embeddings().num_embeddings
    new_vocab = len(tok)
    if new_vocab != old_vocab:
        print(f"[info] resize_token_embeddings: {old_vocab} -> {new_vocab}")
        llm.resize_token_embeddings(new_vocab)
        _tie_if_needed(llm)

    # 3) 特殊 token ID：只在模型缺失时补齐；避免覆盖成错误 ID
    if llm.config.eos_token_id is None and tok.eos_token_id is not None:
        llm.config.eos_token_id = tok.eos_token_id
    if llm.config.pad_token_id is None and tok.pad_token_id is not None:
        llm.config.pad_token_id = tok.pad_token_id
    if llm.config.bos_token_id is None and tok.bos_token_id is not None:
        llm.config.bos_token_id = tok.bos_token_id

    # 4) 最关键：确保权重真正 tying
    if not _check_tying(llm):
        print("[warn] embeddings and lm_head are not tied. Retie now.")
        _tie_if_needed(llm)

    # 5) 包装 MolAware（realtime_mol=False 时需 100% 直通）
    model = MolAwareCausalLM(llm=llm, tokenizer=tok, mol_token=mol_token, debug=False)
    return tok, model

@torch.no_grad()
def generate_once(tok, llm, prompt: str, max_new_tokens=256):
    # 和你的第一段严格对齐：把 inputs 放到同一设备
    dev = next(llm.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)

    out = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,     # 轻微抑制复读
        no_repeat_ngram_size=3,
        use_cache=True,
    )
    # 只取新增的部分
    gen = out[0, inputs["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=False)

@torch.no_grad()
def generate_via_molaware(tok, model, prompt: str, max_new_tokens=256, realtime_mol=False):
    # 直通路径（用于对照验证）
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)
    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        use_cache=True,
        realtime_mol=realtime_mol,
    )
    gen = out[0, inputs["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=False)

if __name__ == "__main__":
    tok, model = load_model(MODEL_PATH, mol_token=MOL_TOKEN)

    prompt = "Ethanol is a common solvent with a density lower than water. Please explain the reason and continue."
    prompt = "What are the common organic solvents?"
    prompt = "which molecule is mor polar? CCO or CCl4?"

    print("\n=== Baseline: LLM direct ===")
    print(generate_once(tok, model.llm, prompt))

    print("\n=== MolAware passthrough (realtime_mol=False) ===")
    print(generate_via_molaware(tok, model, prompt, realtime_mol=True))
