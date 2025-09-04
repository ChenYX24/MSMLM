# modules/mol_aware_lm_simple.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    cumsum = attention_mask.long().cumsum(dim=-1)
    pos = (cumsum - 1).clamp(min=0)
    return pos * attention_mask.long()

class MolAwareCausalLM(nn.Module):
    """
    规则（训练/推理严格一致）：
      - 扫描 <mol>，对每个 <mol> 通过黑盒得到一个向量，按出现顺序“追加到序列末尾”（不是插入原位）。
      - 训练/评估：一次性扩展并喂 inputs_embeds，追加位 labels=-100（不计损）。
      - 推理：逐步生成；实时遇到 <mol> 就先把对应向量作为 inputs_embeds 推进一步（不产词），更新KV，再继续生成。
      - 输出：追加的这些“虚拟步”不会出现在输出 token 序列里（训练忽略损失，推理不计入输出）。
    """

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        mol_token: str = "<mol>",
        debug: bool = False,
        target_layer_for_capture: int = -1,  # 仅用于可选的 hidden 抓取（保留，不启用）
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.mol_token = mol_token
        self.mol_token_id = tokenizer.convert_tokens_to_ids(mol_token)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.debug = debug

        if self.mol_token_id is None or self.mol_token_id < 0:
            raise ValueError(f"Tokenizer does not contain mol_token '{mol_token}'. Please add it first.")

        layers_ref = None
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
            layers_ref = self.llm.model.layers  # LLaMA/LlamaForCausalLM
        elif hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "h"):
            layers_ref = self.llm.transformer.h  # GPT/GPT2 类
        object.__setattr__(self, "_layers_ref", layers_ref)
        self.num_layers = len(self._layers_ref) if self._layers_ref is not None else 0
        self.target_layer_for_capture = (
            self.num_layers - 1 if (target_layer_for_capture < 0 and self.num_layers > 0) else target_layer_for_capture
        )
        self._capture_bucket: List[List[torch.Tensor]] = []
        self._capture_hook = None

    # ---------- 工具：第一层设备 ----------
    def _first_device(self):
        try:
            return self.llm.model.layers[0].input_layernorm.weight.device
        except Exception:
            return next(self.llm.parameters()).device

    # ---------- 黑盒：根据上下文产出要“追加到末尾”的 embedding ----------
    def _black_box_embed_offline(
        self,
        row_ids: torch.Tensor,          # [T]
        row_embeds: torch.Tensor,       # [T, D]
        row_mask: torch.Tensor,         # [T] (1/0)
        pos_mol: int,                   # <mol> 的位置
    ) -> torch.Tensor:
        prev_i = pos_mol - 1
        if prev_i < 0 or (row_mask[prev_i].item() == 0):
            prev_i = pos_mol
        return row_embeds[prev_i]  # [D]

    def _black_box_embed_online(
        self,
        prev_token_id: Optional[int],
        mol_token_id: int,
        embed_tokens: nn.Embedding,
        last_token_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if last_token_embed is not None:
            return last_token_embed
        use_id = prev_token_id if (prev_token_id is not None) else mol_token_id
        return embed_tokens.weight[use_id]

    # ---------- （保留）抓取某层在 mol 位置的 hidden states ----------
    def capture_hidden_at_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_index: Optional[int] = None,
    ) -> List[List[torch.Tensor]]:
        if layer_index is None:
            layer_index = self.target_layer_for_capture
        assert self._layers_ref is not None and 0 <= layer_index < self.num_layers, "Invalid layer for capture"

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long().to(input_ids.device)

        bucket = [[] for _ in range(input_ids.size(0))]

        def _hook(_module, _in, out):
            hidden = out[0] if isinstance(out, tuple) else out  # [B, T, D]
            with torch.no_grad():
                for b in range(hidden.size(0)):
                    mol_pos = (input_ids[b] == self.mol_token_id).nonzero(as_tuple=False).flatten()
                    for p in mol_pos.tolist():
                        if attention_mask[b, p] == 1:
                            bucket[b].append(hidden[b, p].detach().cpu())

        handle = self._layers_ref[layer_index].register_forward_hook(_hook)
        _ = self.llm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        handle.remove()
        return bucket

    # ---------- 训练/评估：一次性扩展 ----------
    def _append_mol_embeds_to_end_offline(
        self,
        input_ids: torch.Tensor,              # [B, T]
        attention_mask: Optional[torch.Tensor],   # [B, T] or None
        labels: Optional[torch.Tensor],           # [B, T] or None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert input_ids.dim() == 2

        embed_tokens = self.llm.get_input_embeddings()
        emb_dev = embed_tokens.weight.device  # ★ embedding 的设备

        # ★ 关键：先把张量搬到 emb_dev，避免 cpu vs cuda 冲突（训练/推理都安全）
        input_ids = input_ids.to(emb_dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(emb_dev)
        if labels is not None:
            labels = labels.to(emb_dev)

        B, T = input_ids.shape
        device = input_ids.device

        embeds = embed_tokens(input_ids)  # [B, T, D]
        D = embeds.size(-1)

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long().to(device)
        has_labels = labels is not None

        rows_embeds, rows_masks, rows_labels = [], [], []
        max_len = 0

        for b in range(B):
            row_ids = input_ids[b]
            row_emb = embeds[b]
            row_msk = attention_mask[b]
            row_lbl = labels[b] if has_labels else None

            new_emb_list = [row_emb[i] for i in range(T)]
            new_msk_list = [int(row_msk[i].item()) for i in range(T)]
            new_lbl_list = [int(row_lbl[i].item()) for i in range(T)] if has_labels else None

            mol_positions = (row_ids == self.mol_token_id).nonzero(as_tuple=False).flatten().tolist()
            for p in mol_positions:
                if new_msk_list[p] == 0:
                    continue
                mol_emb = self._black_box_embed_offline(row_ids, row_emb, row_msk, p)  # [D]
                new_emb_list.append(mol_emb)
                new_msk_list.append(1)
                if has_labels:
                    new_lbl_list.append(-100)

            new_len = len(new_msk_list)
            max_len = max(max_len, new_len)

            new_emb = torch.stack(new_emb_list, dim=0)                     # [T', D]
            new_msk = torch.tensor(new_msk_list, device=device, dtype=row_msk.dtype)  # [T']
            if has_labels:
                new_lbl = torch.tensor(new_lbl_list, device=device, dtype=input_ids.dtype)  # [T']
            else:
                new_lbl = None

            rows_embeds.append(new_emb)
            rows_masks.append(new_msk)
            if has_labels:
                rows_labels.append(new_lbl)

        padded_embeds, padded_masks = [], []
        padded_labels = [] if has_labels else None
        for b in range(B):
            E = rows_embeds[b]; M = rows_masks[b]
            pad_len = max_len - E.size(0)
            if pad_len > 0:
                E = torch.cat([E, torch.zeros(pad_len, D, device=E.device, dtype=E.dtype)], dim=0)
                M = torch.cat([M, torch.zeros(pad_len, device=M.device, dtype=M.dtype)], dim=0)
                if has_labels:
                    L = rows_labels[b]
                    L = torch.cat([L, torch.full((pad_len,), -100, device=L.device, dtype=L.dtype)], dim=0)
                else:
                    L = None
            else:
                L = rows_labels[b] if has_labels else None

            padded_embeds.append(E.unsqueeze(0))
            padded_masks.append(M.unsqueeze(0))
            if has_labels:
                padded_labels.append(L.unsqueeze(0) if L is not None else None)

        new_embeds = torch.cat(padded_embeds, dim=0)   # [B, T_max, D]
        new_masks  = torch.cat(padded_masks, dim=0)    # [B, T_max]
        new_labels = torch.cat(padded_labels, dim=0) if has_labels else None

        if self.debug:
            orig_tokens = attention_mask.sum().item()
            new_tokens  = new_masks.sum().item()
            print(f"[MolAware/offline] appended {int(new_tokens - orig_tokens)} embeddings to batch end")

        return new_embeds, new_masks, new_labels

    # ---------- 前向（训练/评估） ----------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert input_ids is not None, "MolAwareCausalLM 需要 input_ids"

        new_embeds, new_masks, new_labels = self._append_mol_embeds_to_end_offline(input_ids, attention_mask, labels)
        position_ids = build_position_ids(new_masks).to(new_masks.device)

        outputs = self.llm(
            inputs_embeds=new_embeds,
            attention_mask=new_masks,
            position_ids=position_ids,
            labels=new_labels,
            return_dict=True,
            **kwargs,
        )
        return outputs

    # ---------- 推理：逐步，实时插入 ----------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        realtime_mol: bool = True,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        # 统一并强制开启 KV 缓存；避免和 kwargs 冲突
        use_cache = kwargs.pop("use_cache", True)
        try:
            self.llm.config.use_cache = True
        except Exception:
            pass

        # 非实时：直接走底层 HF 的 generate
        if not realtime_mol:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                use_cache=use_cache,
                **kwargs,
            )

        # 实时：支持“输入无 <mol>，输出遇到才触发”
        assert input_ids is not None and input_ids.size(0) == 1, "realtime_mol 仅支持 batch=1"

        llm = self.llm
        dev = self._first_device()
        embed_tokens = llm.get_input_embeddings()
        dtype = embed_tokens.weight.dtype

        # 准备 attention_mask
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

        # (A) 首轮：用 input_ids 前传，建立 past（不传 position_ids，保持与 HF generate 一致）
        outputs = llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        past = outputs.past_key_values
        attn_mask = attention_mask  # 之后在此基础上扩展

        generated_ids: List[int] = []
        last_token_embed: Optional[torch.Tensor] = None
        # 上一个“已确认”token（输入最后一个有效 token）
        prev_token_id: int = int(input_ids[0, attn_mask[0].sum().item() - 1].item())
        end_id = self.eos_token_id if eos_token_id is None else eos_token_id

        # (B) 自回归主循环
        for _ in range(max_new_tokens):
            logits = outputs.logits[:, -1, :]  # [1, vocab]

            # 轻度防复读：仅对已生成过的 token 降权
            if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
                uniq = list(set(generated_ids))
                logits[:, uniq] = logits[:, uniq] / repetition_penalty

            # 采样 / 贪心
            if do_sample:
                if temperature != 1.0:
                    logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = probs.cumsum(dim=-1)
                    cutoff = (cumprobs > top_p).float().cumsum(dim=-1).bool()
                    sorted_logits[cutoff] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_id = int(next_token.item())

            # 命中 <mol> ：执行“embedding-only”一步（不产词，不计入 generated_ids）
            if next_id == self.mol_token_id:
                insert_vec = self._black_box_embed_online(
                    prev_token_id=prev_token_id,
                    mol_token_id=self.mol_token_id,
                    embed_tokens=embed_tokens,
                    last_token_embed=last_token_embed,
                ).to(dev, dtype=dtype)

                # 扩一位 mask；不显式给 position_ids，交给模型依据 mask 自动推断
                attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
                step_emb = insert_vec.view(1, 1, -1)

                outputs = llm(
                    inputs_embeds=step_emb,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=use_cache,
                    return_dict=True,
                    **kwargs,
                )
                past = outputs.past_key_values
                last_token_embed = insert_vec
                # 不更新 prev_token_id（因为没真正产词）
                continue

            # 正常产词一步：input_ids + cache
            step_ids = torch.tensor([[next_id]], device=dev, dtype=input_ids.dtype)
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)

            outputs = llm(
                input_ids=step_ids,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=use_cache,
                return_dict=True,
                **kwargs,
            )
            past = outputs.past_key_values

            generated_ids.append(next_id)
            prev_token_id = next_id
            last_token_embed = embed_tokens(step_ids)[0, 0, :]

            if end_id is not None and next_id == end_id:
                break

        if not generated_ids:
            return input_ids  # 没新增

        gen = torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)
        return torch.cat([input_ids, gen], dim=1)

    @property
    def config(self):
        return getattr(self.llm, "config", None)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        if self.config is not None:
            try:
                self.config.use_cache = False
            except Exception:
                pass
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs is not None:
                    return self.llm.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
                if kwargs:
                    return self.llm.gradient_checkpointing_enable(**kwargs)
                return self.llm.gradient_checkpointing_enable()
            except TypeError:
                return self.llm.gradient_checkpointing_enable()
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            try:
                out = self.llm.gradient_checkpointing_disable()
            except TypeError:
                out = None
        else:
            out = None
        if self.config is not None:
            try:
                self.config.use_cache = True
            except Exception:
                pass
        return out

    @staticmethod
    def _storage_id(t):
        try: return t.untyped_storage().data_ptr()
        except Exception: return t.storage().data_ptr()

    def state_dict(self, *args, **kwargs):
        sd = self.llm.state_dict(*args, **kwargs)  # 只导出底层
        seen = {}
        for k, v in list(sd.items()):
            if not isinstance(v, torch.Tensor): continue
            sid = self._storage_id(v)
            if sid in seen:
                sd[k] = v.clone()
            else:
                seen[sid] = k
        return sd
