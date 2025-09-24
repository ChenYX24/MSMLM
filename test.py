# plain_llm_infer_wrapper.py
# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional, Dict, Any, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


RoleMsg = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class PlainLLMGenerator:
    """
    统一封装：
      - load(config) -> 初始化 tokenizer + LLM
      - generate(prompt|messages, ...) -> 文本或 token id 输出

    关键改动：
      * 自动检测并使用 tokenizer 的 chat_template（若存在）
      * 支持 messages=[{role, content}, ...]；若仅给 prompt 也会自动包成 <user>
      * 若无 chat_template，回退到简易模板：<|system|>...<|user|>...<|assistant|>
      * 默认恢复 add_special_tokens 语义，避免你之前关闭后导致的降质
    """

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.loaded_cfg: Dict[str, Any] = {}
        self._has_chat_template: bool = False

    # ----------------- 内部工具 -----------------
    def _detect_chat_template(self) -> bool:
        """
        检测是否存在 chat_template。
        大部分 Instruct/Chat 模型会在 tokenizer_config.json 里带有 "chat_template"。
        """
        try:
            tmpl = getattr(self.tokenizer, "chat_template", None)
            return bool(tmpl and isinstance(tmpl, str) and tmpl.strip())
        except Exception:
            return False

    def _wrap_chat_with_template(
        self,
        messages: List[RoleMsg],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        优先使用 tokenizer.apply_chat_template；失败则使用手工兜底模板。
        返回：已拼接好的“带特殊标记”的字符串（此时再 encode 时请使用 add_special_tokens=False）。
        """
        assert self.tokenizer is not None
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            # 简易兜底（ChatML/Llama3风格混合；足够安全通用）
            buf = []
            for m in messages:
                role = m.get("role", "user").strip().lower()
                content = m.get("content", "")
                if role == "system":
                    buf.append(f"<|system|>{content}\n")
                elif role == "assistant":
                    buf.append(f"<|assistant|>{content}\n")
                else:
                    buf.append(f"<|user|>{content}\n")
            if add_generation_prompt:
                buf.append("<|assistant|>")
            return "".join(buf)

    def _coerce_to_messages(
        self,
        prompt_or_messages: Union[str, List[RoleMsg]],
        system_prompt: Optional[str] = None,
    ) -> List[RoleMsg]:
        """
        将输入统一成 messages 列表。
        """
        if isinstance(prompt_or_messages, str):
            msgs: List[RoleMsg] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt_or_messages})
            return msgs
        # 已经是 messages
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + prompt_or_messages
        return prompt_or_messages

    # ----------------- 对外接口 -----------------
    def load(self, cfg: Dict[str, Any]) -> None:
        """
        cfg 示例：
        {
          "ckpt_dir": "...",
          "device": "cuda:0",
          "dtype": "bf16" | "fp32"
        }
        """
        self.loaded_cfg = cfg
        ckpt_dir = cfg["ckpt_dir"]
        self.device = cfg.get("device", self.device)

        # ---- 加载 tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            # 有些基座没 pad_token，兜底用 eos
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 记录是否有 chat_template
        self._has_chat_template = self._detect_chat_template()
        if self._has_chat_template:
            logging.info("✅ Detected tokenizer.chat_template — chat prompting enabled.")
        else:
            logging.info("ℹ️ No chat_template in tokenizer — will use a safe manual fallback template.")

        # ---- 精度 ----
        dtype_flag = str(cfg.get("dtype", "bf16")).lower()
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and "bf16" in dtype_flag) else torch.float32

        # ---- 加载 LLM ----
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch_dtype,
            device_map=None,  # 也可以设 "auto"
        ).to(self.device)

        # 同步 id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id

        logging.info(f"✅ Plain LLM loaded from {ckpt_dir} on {self.device}.")

    @torch.no_grad()
    def generate(
        self,
        prompt_or_messages: Union[str, List[RoleMsg]],
        *,
        system_prompt: Optional[str] = None,
        use_chat_template: Optional[bool] = None,  # None=auto(有则用), True=强制用, False=不用
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        eos_token_id: Optional[int] = None,
        skip_special_tokens: bool = True,
        return_ids: bool = False,
        add_special_tokens_when_no_template: bool = True,
    ):
        """
        - prompt_or_messages: 可以是 str（当作 user）或 messages 列表([{"role":..., "content":...}, ...])
        - system_prompt: 需要时可传
        - use_chat_template:
            * None：自动（若 tokenizer 有模板则用）
            * True：强制使用模板（若无则回退手工模板）
            * False：不使用模板，直接把字符串送进模型
        - add_special_tokens_when_no_template:
            * 当不用模板时，是否在 encode 时开启 add_special_tokens（默认 True，更安全）
        """
        assert self.model is not None and self.tokenizer is not None, "请先调用 load(config) 完成初始化。"

        if use_chat_template is None:
            use_chat_template = self._has_chat_template

        # --- 构造输入文本 ---
        if use_chat_template:
            messages = self._coerce_to_messages(prompt_or_messages, system_prompt)
            rendered = self._wrap_chat_with_template(messages, add_generation_prompt=True)
            # 模板已经注入特殊标记 -> encode 时不要再加 special tokens
            enc = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        else:
            # 不用模板：若给的是 messages，拼成简单串
            if isinstance(prompt_or_messages, list):
                # 简单连接成： [system]\n[user]\n[assistant:空等待续写]
                rendered = self._wrap_chat_with_template(
                    self._coerce_to_messages(prompt_or_messages, system_prompt),
                    add_generation_prompt=True,
                )
                enc = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            else:
                rendered = prompt_or_messages if system_prompt is None else (system_prompt + "\n\n" + prompt_or_messages)
                enc = self.tokenizer(
                    rendered,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens_when_no_template,  # 默认 True：更稳
                )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        out_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
        )

        out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=skip_special_tokens)
        return (out_text, out_ids) if return_ids else out_text


if __name__ == "__main__":
    CONFIG = {
        # "ckpt_dir": "/data1/chenyuxuan/Project/MSMLM/model/llama3.2-chem-sft-gnn/llm_gnn_nofreeze2/checkpoint-1000",
        "ckpt_dir": "/data1/lvchangwei/LLM/llama3.2-cpt/llama3.2-instruct-cptv1/v9-20250919-113647/checkpoint-185",  # 换成你的普通 LLM 路径
        "device": "cuda:0",
        "dtype": "bf16",
    }

    gen = PlainLLMGenerator()
    gen.load(CONFIG)

    # 用“messages + 自动模板”
    messages = [
        {"role": "user", "content": "请根据 SMILES: CC1N2CC1(C2)N1CC1 定性分析其量子化学性质。"},
    ]
    text1 = gen.generate(
        messages,
        use_chat_template=None,   # None=自动（若检测到模板则用）
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.05,
    )
    print("\n=== Chat-template (auto) ===")
    print(text1)

    # 用“纯字符串 + 强制不用模板（不推荐，但可用于基座 or 自控）”
    prompt = "请根据 SMILES: CC1N2CC1(C2)N1CC1 定性分析其量子化学性质。"
    text2 = gen.generate(
        prompt,
        use_chat_template=False,              # 不使用模板
        add_special_tokens_when_no_template=True,  # 没模板时，建议开 True
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.05,
    )
    print("\n=== No-template (raw prompt) ===")
    print(text2)
    
    msgs = [
        {"role": "user", "content": "请根据 SMILES: CC1N2CC1(C2)N1CC1 定性分析其量子化学性质。"}
    ]

    text = gen.generate(
        msgs,
        use_chat_template=None,               # 自动检测，如有模板就用
        do_sample=False,                      # 关闭采样更稳定
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=400,
    )
    print(text)
