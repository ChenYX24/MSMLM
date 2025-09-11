# inference_multigpu.py
# -*- coding: utf-8 -*-
import os
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from modules.mol_aware_lm_simple import MolAwareCausalLM
from transformers.generation import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

def build_prompt(tokenizer, user_text: str) -> str:
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl:  # 非空才用
        try:
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"[warn] apply_chat_template failed: {e}; fallback to manual template.")

    # —— 回退：与你训练一致的模板（你已经在 tokenizer 里添加了 <|user|> / <|assistant|>）——
    return f"<|user|>\n{user_text}\n<|assistant|>\n"


def load_tokenizer(cfg):
    llm_name = cfg["paths"]["llm_name_or_path"]
    tok = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    role_specials = ["<|user|>", "<|assistant|>"]
    to_add = []
    mol_token = cfg["tokens"]["mol_token"]
    if mol_token not in tok.get_vocab():
        to_add.append(mol_token)
    for t in role_specials:
        if t not in tok.get_vocab():
            to_add.append(t)
    if to_add:
        tok.add_special_tokens({"additional_special_tokens": to_add})
    return tok


def pick_model_path(cfg, override_path=None):
    if override_path:
        return override_path
    out_dir = cfg["paths"].get("output_dir", "")
    if out_dir and os.path.isdir(out_dir) and any(
        os.path.exists(os.path.join(out_dir, f))
        for f in ["pytorch_model.bin", "model.safetensors", "adapter_model.bin"]
    ):
        return out_dir
    return cfg["paths"]["llm_name_or_path"]


def build_model_and_wrap(cfg, tokenizer, model_path=None):
    model_path = pick_model_path(cfg, model_path)
    dtype = torch.bfloat16 if cfg["train"].get("bf16", False) else torch.float32

    # 1) 先在 CPU 加载，避免已分片导致 resize 失效
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,              # <—— 先不要自动分配
        low_cpu_mem_usage=True,
    )

    # 2) 明确 resize 到 len(tokenizer)，并同步 config
    new_size = len(tokenizer)
    old_in = llm.get_input_embeddings().num_embeddings
    if old_in != new_size:
        llm.resize_token_embeddings(new_size)
        llm.config.vocab_size = new_size

    # 3) 确认输出头（lm_head）尺寸一致；有些模型不会自动改
    out = llm.get_output_embeddings()
    if out is not None and getattr(out, "out_features", None) != new_size:
        with torch.no_grad():
            old_head = out
            in_features = old_head.in_features
            bias = old_head.bias is not None
            new_head = torch.nn.Linear(in_features, new_size, bias=bias)
            # 可选：拷贝旧权重到前 old_size 区
            old_size = old_head.out_features
            new_head.weight[:old_size].copy_(old_head.weight)
            if bias:
                new_head.bias[:old_size].copy_(old_head.bias)
        # 注意：根据你的模型实际属性名设置
        llm.lm_head = new_head
        try:
            llm.tie_weights()
        except Exception:
            pass

    # 4) 再分配到多卡
    # 简易做法：单机多卡自动切分（需要 accelerate>=0.21）
    llm = llm.to(dtype)  # 确保 dtype 正确
    # llm = llm.to("cuda") # 如果是单卡就这样；多卡建议：
    from accelerate import dispatch_model, infer_auto_device_map
    device_map = infer_auto_device_map(llm, no_split_module_classes=["LlamaDecoderLayer"])
    llm = dispatch_model(llm, device_map=device_map)

    wrapper = MolAwareCausalLM(
        llm=llm,
        tokenizer=tokenizer,
        mol_token=cfg["tokens"]["mol_token"],
        target_layer=cfg.get("inject", {}).get("target_layer", -1),
        debug=False,
    )
    wrapper.eval()

    # 5) 打印自检，确保真的成功
    print("[check] len(tokenizer) =", len(tokenizer))
    print("[check] input_emb.num_embeddings =", wrapper.llm.get_input_embeddings().num_embeddings)
    out = wrapper.llm.get_output_embeddings()
    print("[check] lm_head.out_features =", getattr(out, "out_features", None))

    return wrapper


class _PatchLLMForward:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.llm = wrapper.llm
        self._orig_forward = None

    def __enter__(self):
        self._orig_forward = self.llm.forward

        def _wrapped_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            self.wrapper._curr_input_ids = input_ids
            try:
                return self._orig_forward(*args, **kwargs)
            finally:
                self.wrapper._curr_input_ids = None

        self.llm.forward = _wrapped_forward
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_forward is not None:
            self.llm.forward = self._orig_forward
        self._orig_forward = None


def contains_mol(text: str, mol_token: str) -> bool:
    return mol_token in text


def build_generation_config(tokenizer, use_cache: bool, max_new_tokens=256,
                            temperature=0.7, top_p=0.9, do_sample=True):
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        use_cache=use_cache,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )



class SanitizeLogits(LogitsProcessor):
    def __init__(self, min_val=-1e4, max_val=1e4):
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=self.max_val, neginf=self.min_val)
        return scores.clamp_(min=self.min_val, max=self.max_val)

def generate_text(model, tokenizer, prompt, **kwargs):
    """
    prompt: 用户原始输入（纯文本）
    其余采样参数从 kwargs 取（temperature/top_p/top_k/max_new_tokens/greedy 等）
    """
    # —— 参数整理（含默认值与保护）——
    greedy = bool(kwargs.pop("greedy", False))
    do_sample = not greedy
    temperature = max(1e-3, float(kwargs.pop("temperature", 1.0)))
    top_p = float(kwargs.pop("top_p", 0.95))
    top_k = int(kwargs.pop("top_k", 50))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 256))

    # 从 kwargs 去掉不会被 generate 接受的自定义键（避免 TypeError）
    mol_token = kwargs.pop("mol_token", None)

    # —— 构造“干净”的提示 ——（避免手写 <|begin_of_text|> 混入）
    chat_prompt = build_prompt(tokenizer, prompt)

    # —— tokenize —— 
    enc = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # —— 禁止不该生成的特殊 token —— 
    processors = LogitsProcessorList([SanitizeLogits()])

    # —— 终止符（兼容 <|eot_id|>）——
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != -1:
        eos_ids.append(eot_id)

    # —— use_cache 策略：含 <mol> 的样本默认关 cache，让中层改写影响后续 token —— 
    # （如果你想全局关/开，也可以在 main 里显式传 use_cache）
    use_cache = kwargs.pop("use_cache", None)
    if use_cache is None:
        use_cache = False if (mol_token and contains_mol(chat_prompt, mol_token)) else True

    # —— 生成 ——（为了稳定，禁用 autocast）
    with torch.autocast(device_type="cuda", enabled=False):
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logits_processor=processors,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids if eos_ids else tokenizer.eos_token_id,
        )

    # 仅解码“新生成”的部分
    gen_ids = out_ids[0, input_ids.size(1):]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="/data1/chenyuxuan/Project/MSMLM/code/mol_sft/configs/config.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--user", default="Ethanol is a common solvent with a density lower than water. Please explain the reason and continue.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = load_tokenizer(cfg)
    model = build_model_and_wrap(cfg, tokenizer, model_path=args.model_path)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.user,
        mol_token=cfg["tokens"]["mol_token"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )
    print("\n=== Generation ===\n")
    print(text)


if __name__ == "__main__":
    main()
