# mol_aware_infer_wrapper.py
# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer

logging.getLogger("rdkit").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from modules.mol_aware_lm import MolAwareCausalLM
import re


class MolAwareGenerator:
    """
    统一封装：load(config) -> 初始化 tokenizer + MolAwareCausalLM
             generate(prompt, ...) -> 文本或 token id 输出
    """

    def __init__(self):
        self.model: Optional[MolAwareCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.loaded_cfg: Dict[str, Any] = {}

    # ------------------------ 内部工具 ------------------------
    def _ensure_special_tokens(self):
        """
        只校验特殊 token 是否存在；推理期禁止新增，以免撕裂 embedding 权重。
        同时对 eos/pad 做“仅设置 id”的兜底（不改词表）。
        """
        assert self.tokenizer is not None
        vocab = self.tokenizer.get_vocab()

        needed = ["<mol>", "<|user|>", "<|assistant|>"]
        missing = [t for t in needed if t not in vocab]
        if missing:
            raise RuntimeError(
                f"[vocab-mismatch] 推理期禁止新增 token。缺失: {missing}。"
                f"请确保导出的 tokenizer 已包含这些 token（训练/拆分阶段对齐）。"
            )

        # ====== 兜底 eos/bos/pad（不改词表，仅设置已有 id）======
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id >= 0:
                # 优先用 Llama3 的 eot 作为 eos
                self.tokenizer.eos_token = "<|eot_id|>"

        if getattr(self.tokenizer, "eos_token_id", None) is None:
            try_ids = [
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "sep_token_id", None),
                getattr(self.tokenizer, "cls_token_id", None),
                getattr(self.tokenizer, "bos_token_id", None),
            ]
            try_ids = [t for t in try_ids if t is not None]
            self.tokenizer.eos_token_id = try_ids[0] if try_ids else 0

        if self.tokenizer.pad_token is None:
            # 只把 pad 对齐到已有 eos，不新增词表
            if isinstance(self.tokenizer.eos_token, str):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _sync_vocab_and_embeddings(self, strict: bool = True):
        """
        校验 tokenizer 与模型的词表大小是否一致。
        - strict=True: 不一致就报错（推荐，防止隐性扩表）
        - strict=False: 若 tokenizer 更大，则扩 model embedding（不改 tokenizer）
        """
        assert self.model is not None and self.tokenizer is not None
        v_tok = len(self.tokenizer)
        v_model = self.model.llm.get_input_embeddings().weight.size(0)

        if v_tok == v_model:
            # 轻同步几个 id（不会引起权重形状变化）
            self.model.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.llm.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.llm.config.bos_token_id = self.tokenizer.bos_token_id
            return

        if strict:
            raise RuntimeError(
                f"[vocab-mismatch] tokenizer({v_tok}) != model-emb({v_model})。"
                f"请在训练/拆分阶段确保词表一致，推理期不要扩表。"
            )
        else:
            if v_tok > v_model:
                self.model.llm.resize_token_embeddings(v_tok)
                self.model.llm.config.vocab_size = v_tok
                logging.info(f"[warn] Resized model embeddings from {v_model} -> {v_tok} (strict=False)")
            else:
                raise RuntimeError(
                    f"[vocab-mismatch] tokenizer({v_tok}) < model-emb({v_model})，"
                    f"请加载与模型匹配的 tokenizer。"
                )

            self.model.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.llm.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.llm.config.bos_token_id = self.tokenizer.bos_token_id

    # ------------------------ 对外 API ------------------------
    def load(self, cfg: Dict[str, Any]) -> None:
        """
        cfg 示例：
        {
          "ckpt_dir": "...",
          "device": "cuda:0",
          "dtype": "bf16" | "fp32",
          "add_safe_globals": false,
          "debug": true,
          "diffusion": {...},
          "diffusion_adapter": {...},
          "diffusion_generation": {...}
        }
        """
        self.loaded_cfg = cfg
        ckpt_dir = cfg["ckpt_dir"]
        self.device = cfg.get("device", self.device)

        # ---- PyTorch 2.6: 可选安全反序列化（通常推理不需要）----
        if cfg.get("add_safe_globals", False):
            import torch.serialization as ts
            ts.add_safe_globals([np.core.multiarray._reconstruct])

        # ---- 加载 tokenizer（来自 ckpt_dir；该目录应包含 tokenizer 文件）----
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self._ensure_special_tokens()

        # ---- 精度 ----
        dtype_flag = str(cfg.get("dtype", "bf16")).lower()
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and "bf16" in dtype_flag) else torch.float32

        # ---- diffusion 配置（可选）----
        diffusion_config = cfg.get("diffusion", None)
        diffusion_adapter_config = cfg.get("diffusion_adapter", None)

        # ---- 加载组合模型 ----
        # ckpt_dir 可以是“拆分后的根目录”（含 molaware_metadata.json、llm/、extras/）
        # 也可以是训练时的合并目录（只要 from_pretrained 能找到 metadata）
        self.model = MolAwareCausalLM.from_pretrained(
            save_directory=ckpt_dir,
            tokenizer=self.tokenizer,
            diffusion_config=diffusion_config,
            diffusion_adapter_config=diffusion_adapter_config,
            torch_dtype=torch_dtype,
            device_map=None,  # 需要也可改为 'auto'
        ).to(self.device)

        self.model.debug = bool(cfg.get("debug", False))

        # 生成时的扩散参数（如果你的类支持）
        gen_conf = cfg.get("diffusion_generation", {})
        if hasattr(self.model, "diffusion_gen_num_nodes_lig"):
            self.model.diffusion_gen_num_nodes_lig = gen_conf.get("num_nodes_lig", None)

        # ---- 关键：严格校验 tokenizer/embedding 一致性（不扩表）----
        self._sync_vocab_and_embeddings(strict=True)

        logging.info(f"✅ Model & tokenizer loaded from {ckpt_dir} on {self.device}.")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        add_dialog_wrapper: bool = True,          # 默认套上 chat 模板
        realtime_mol: bool = True,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        skip_special_tokens: bool = True,
        eos_token_id: Optional[int] = None,
        return_ids: bool = False,
        stop_on_user_token: bool = True,
    ):
        """
        - add_dialog_wrapper: 使用 tokenizer 的 chat template（system+user）
        - stop_on_user_token: 生成到下一轮 <|user|> 或 <|eot_id|> 或 </s> 时截断
        """
        assert self.model is not None and self.tokenizer is not None, "请先调用 load(config) 完成初始化。"

        # ====== 用 chat template 封装 ======
        SYSTEM_MSG = "You are a careful chemist. Follow the requested output format exactly."

        def _fallback_manual_chat(u: str, s: str = SYSTEM_MSG) -> str:
            # 与 SFT 的 <|user|>/<|assistant|> 模板一致
            return f"<|user|>{s}\n<|assistant|>OK\n<|user|>{u}\n<|assistant|>"

        if add_dialog_wrapper:
            try:
                text_in = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text_in = _fallback_manual_chat(prompt)
        else:
            text_in = prompt

        enc = self.tokenizer(text_in, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        v_tok = len(self.tokenizer)
        ids_max = int(input_ids.max().item())
        assert ids_max < v_tok, f"input_ids 最大值 {ids_max} >= tokenizer 词表 {v_tok}，请检查 tokenizer/embedding 同步。"

        # ====== 构造 eos / stop tokens ======
        eos_id = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        if eos_id is None or eos_id < 0:
            eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            eos_id = eot if (eot is not None and eot >= 0) else None

        # 便于早停
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # ====== 真正生成 ======
        out_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            realtime_mol=realtime_mol,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_id,
            repetition_penalty=repetition_penalty,
        )

        # 只取“新生成”的 tokens（不连同 prompt）
        prompt_len = input_ids.shape[1]
        gen_ids = out_ids[0, prompt_len:]

        # ====== 早停（仅在生成段内截断） ======
        if stop_on_user_token:
            try:
                user_tok_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
                cut_pos = None
                gen_list = gen_ids.tolist()

                if eot_id is not None and eot_id in gen_list:
                    cut_pos = gen_list.index(eot_id)
                if cut_pos is None and user_tok_id is not None and user_tok_id in gen_list:
                    cut_pos = gen_list.index(user_tok_id)
                if cut_pos is not None:
                    gen_ids = gen_ids[:cut_pos]
            except Exception:
                pass

        # ====== 只解码 assistant 段 ======
        raw_text = self.tokenizer.decode(
            gen_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True
        )

        # 兼容“手写模板”时，保留 <|assistant|> 之后的内容
        assistant_text = raw_text.split("<|assistant|>")[-1].lstrip()

        # ====== 检测并标记 <mol> ======
        # token 级检测（更可靠）：看生成的 tokens 是否含有 <mol> 的 id
        mol_tok_id = self.tokenizer.convert_tokens_to_ids("<mol>")
        token_has_mol = (mol_tok_id is not None and mol_tok_id in gen_ids.tolist())

        single_pattern = re.compile(r"</?mol\s*>", flags=re.IGNORECASE)

        mol_spans = []

        text_has_mol = bool(mol_spans) or bool(single_pattern.search(assistant_text))
        has_mol = token_has_mol or text_has_mol



        result = {
            "text": assistant_text,
            "has_mol": bool(has_mol),
            "mol_count": len(mol_spans) if mol_spans else (1 if has_mol else 0),
            "mol_spans": mol_spans,  
        }

        return assistant_text



if __name__ == "__main__":
    CONFIG = {
        "ckpt_dir": "/data1/chenyuxuan/Project/MSMLM/model/llama3.2-chem-sft-gnn/llm_gnn_nofreeze_split/checkpoint-1000",
        "device": "cuda:0",
        "dtype": "bf16",
        "add_safe_globals": False,
        "debug": True,

        "diffusion": {
            "checkpoint_path": "/data1/chenyuxuan/Project/MSMLM/model/diffusion/pubchem_fullatom_cond_0806_v1/log/pubchem_fullatom_cond_0806_v1/checkpoints/last-v1.ckpt",
            "device": "cuda:0",
            "cond_dim": 3072,
        },
        "diffusion_adapter": {
            # "ckpt_path": "/data1/chenyuxuan/Project/MSMLM/model/diffusion_mlp/ckpt/diffusion_mlp_epoch1.pt"
            "ckpt_path":"/data1/chenyuxuan/Project/MSMLM/model/llama3.2-chem-sft-gnn/llm_gnn_nofreeze_split/checkpoint-1000/extras/diffusion_adapter.pt"
        },
        "diffusion_generation": {
            "num_nodes_lig": None
        },
    }

    gen = MolAwareGenerator()
    gen.load(CONFIG)

    prompt = "Provided the molecule with the SMILES CC1N2CC1(C2)N1CC1 below, what are some of its quantum chemical properties?"
    text = gen.generate(
        prompt,
        add_dialog_wrapper=True,
        realtime_mol=False,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.05,
        skip_special_tokens=True,
    )
    print("\n=== Generated Text ===")
    print(text)
