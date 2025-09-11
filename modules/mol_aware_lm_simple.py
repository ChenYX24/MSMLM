# mol_aware_lm_integrated.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
import logging

# 导入你提供的原始模块
from .gnn import GVPEncoder, ATOM_TYPES
from .mlp import MLPAdapter
from .tools import extract_and_convert_online

# 禁用RDKit日志
logging.getLogger("rdkit").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    cumsum = attention_mask.long().cumsum(dim=-1)
    pos = (cumsum - 1).clamp(min=0)
    return pos * attention_mask.long()

# --- Diffusion 管道占位符 ---
class DummyDiffusionEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.proj = nn.Linear(200, output_dim)
    def forward(self, llm_context: str) -> torch.Tensor:
        # 这个方法仅用于占位和演示，你将在这里实现你的 Diffusion 管道
        logging.info("Using Dummy Diffusion Encoder as fallback.")
        embedding = torch.randn(200, device=self.proj.weight.device)
        return self.proj(embedding)

# --- 原始 MolAwareCausalLM 类的修改版本 ---
class MolAwareCausalLM(nn.Module):
    """
    修改后的 MolAwareCausalLM，集成了 NER、GNN 和 Diffusion 管道。
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
        proxy: Optional[str] = None,
        debug: bool = False,
        target_layer_for_capture: int = -1,
        gvp_encoder_config: Optional[Dict] = None,
        mol_adapter_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.mol_token = mol_token
        self.mol_token_id = tokenizer.convert_tokens_to_ids(mol_token)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.debug = debug
        self.proxy = proxy

        if self.mol_token_id is None or self.mol_token_id < 0:
            raise ValueError(f"Tokenizer does not contain mol_token '{mol_token}'. Please add it first.")

        layers_ref = None
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
            layers_ref = self.llm.model.layers
        elif hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "h"):
            layers_ref = self.llm.transformer.h
        object.__setattr__(self, "_layers_ref", layers_ref)
        self.num_layers = len(self._layers_ref) if self._layers_ref is not None else 0
        self.target_layer_for_capture = (
            self.num_layers - 1 if (target_layer_for_capture < 0 and self.num_layers > 0) else target_layer_for_capture
        )
        self._capture_bucket: List[List[torch.Tensor]] = []
        self._capture_hook = None
        
        # --- 1. init: 添加 gvp_encoder, mol_adapter 和 diffusion_encoder ---
        llm_hidden_size = self.llm.config.hidden_size
        
        # GVPEncoder 的配置
        gvp_encoder_cfg = {
            "node_dims": (10, 1),
            "edge_dims": (1, 1), 
            "hidden_scalar_dim": 256,
            "hidden_vector_dim": 16,
            "output_dim": 256,
            "num_layers": 4,
        }
        if gvp_encoder_config:
            gvp_encoder_cfg.update(gvp_encoder_config)
        
        # MLPAdapter 的配置
        mol_adapter_cfg = {
            "input_dim": gvp_encoder_cfg["output_dim"],
            "output_dim": llm_hidden_size,
            "hidden_dim": llm_hidden_size // 2,
            "num_layers": 2,
        }
        if mol_adapter_config:
            mol_adapter_cfg.update(mol_adapter_config)
            
        self.gvp_encoder = GVPEncoder(**gvp_encoder_cfg).to(self._first_device())
        self.mol_adapter = MLPAdapter(**mol_adapter_cfg).to(self._first_device())
        self.diffusion_encoder = DummyDiffusionEncoder(llm_hidden_size).to(self._first_device())
        
        self.smiles_cache: Dict[str, str] = {}
        
    def _first_device(self):
        try:
            return self.llm.model.layers[0].input_layernorm.weight.device
        except Exception:
            return next(self.llm.parameters()).device

    def _get_smiles_from_context(self, llm_context: str) -> Optional[str]:
        if llm_context in self.smiles_cache:
            smiles_map = self.smiles_cache[llm_context]
        else:
            smiles_map = extract_and_convert_online(llm_context, self.proxy)
            self.smiles_cache[llm_context] = smiles_map
            
        if not smiles_map:
            return None
        
        last_cem = ""
        last_idx = -1
        for cem_name in smiles_map:
            idx = llm_context.rfind(cem_name)
            if idx > last_idx:
                last_idx = idx
                last_cem = cem_name
        
        return smiles_map.get(last_cem)

    def _black_box_embed_offline(
        self,
        row_ids: torch.Tensor,
        row_embeds: torch.Tensor,
        row_mask: torch.Tensor,
        pos_mol: int,
    ) -> torch.Tensor:
        """
        训练/评估时，根据上下文 SMILES 走 GVP，否则走 Diffusion。
        """
        llm_context = self.tokenizer.decode(row_ids[:pos_mol].tolist(), skip_special_tokens=True)
        smiles = self._get_smiles_from_context(llm_context)
        
        if smiles:
            logging.info(f"✅ (Offline) Found SMILES for context. Using GVP pipeline.")
            gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
            mol_embedding = self.mol_adapter(gvp_embedding)
            return mol_embedding
        else:
            logging.warning(f"❌ (Offline) No SMILES found. Using Diffusion pipeline as fallback.")
            return self.diffusion_encoder(llm_context)

    def _black_box_embed_online(
        self,
        llm_context_text: str,
    ) -> torch.Tensor:
        """
        推理时，根据上下文 SMILES 走 GVP，否则走 Diffusion。
        （这里简化了，直接传入完整上下文文本）
        """
        smiles = self._get_smiles_from_context(llm_context_text)

        if smiles:
            logging.info(f"✅ (Online) Found SMILES for context. Using GVP pipeline.")
            gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
            mol_embedding = self.mol_adapter(gvp_embedding)
            return mol_embedding
        else:
            logging.warning(f"❌ (Online) No SMILES found. Using Diffusion pipeline as fallback.")
            return self.diffusion_encoder(llm_context_text)

    # ---------- 前向（训练/评估） ----------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 此方法与你原代码基本一致，但 _append_mol_embeds_to_end_offline
        # 将调用我们新定义的 _black_box_embed_offline
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

    def _append_mol_embeds_to_end_offline(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert input_ids.dim() == 2
        embed_tokens = self.llm.get_input_embeddings()
        emb_dev = embed_tokens.weight.device
        input_ids = input_ids.to(emb_dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(emb_dev)
        if labels is not None:
            labels = labels.to(emb_dev)
        
        B, T = input_ids.shape
        device = input_ids.device
        embeds = embed_tokens(input_ids)
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
                
                # ★ 调用新的黑盒函数
                mol_emb = self._black_box_embed_offline(row_ids, row_emb, row_msk, p)
                new_emb_list.append(mol_emb)
                new_msk_list.append(1)
                if has_labels:
                    new_lbl_list.append(-100)
            
            new_len = len(new_msk_list)
            max_len = max(max_len, new_len)
            new_emb = torch.stack(new_emb_list, dim=0)
            new_msk = torch.tensor(new_msk_list, device=device, dtype=row_msk.dtype)
            new_lbl = torch.tensor(new_lbl_list, device=device, dtype=input_ids.dtype) if has_labels else None
            
            rows_embeds.append(new_emb)
            rows_masks.append(new_msk)
            if has_labels:
                rows_labels.append(new_lbl)
        
        # Padding
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
        
        new_embeds = torch.cat(padded_embeds, dim=0)
        new_masks = torch.cat(padded_masks, dim=0)
        new_labels = torch.cat(padded_labels, dim=0) if has_labels else None
        
        if self.debug:
            orig_tokens = attention_mask.sum().item()
            new_tokens = new_masks.sum().item()
            print(f"[MolAware/offline] appended {int(new_tokens - orig_tokens)} embeddings to batch end")
            
        return new_embeds, new_masks, new_labels

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
        use_cache = kwargs.pop("use_cache", True)
        try:
            self.llm.config.use_cache = True
        except Exception:
            pass
            
        if not realtime_mol:
            # 非实时，走原始逻辑
            return super().generate(
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

        assert input_ids is not None and input_ids.size(0) == 1, "realtime_mol 仅支持 batch=1"
        
        llm = self.llm
        dev = self._first_device()
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        
        outputs = llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        past = outputs.past_key_values
        attn_mask = attention_mask
        generated_ids: List[int] = []
        end_id = self.eos_token_id if eos_token_id is None else eos_token_id
        
        for _ in range(max_new_tokens):
            logits = outputs.logits[:, -1, :]
            
            # Repetition penalty logic...
            if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
                uniq = list(set(generated_ids))
                logits[:, uniq] = logits[:, uniq] / repetition_penalty
            
            # Sampling / Greedy logic...
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
            
            if next_id == self.mol_token_id:
                # 获取当前完整的上下文
                current_context_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=dev)], dim=1)
                llm_context_text = self.tokenizer.decode(current_context_ids[0].tolist(), skip_special_tokens=True)
                
                # ★ 调用新的黑盒函数
                mol_embedding = self._black_box_embed_online(llm_context_text)
                
                outputs = llm(
                    inputs_embeds=mol_embedding.view(1, 1, -1),
                    attention_mask=torch.cat([attn_mask, torch.ones(1, 1, device=dev)], dim=1),
                    past_key_values=past,
                    use_cache=use_cache,
                    return_dict=True,
                    **kwargs,
                )
                past = outputs.past_key_values
                attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
                continue
            
            step_ids = next_token
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
            
            if end_id is not None and next_id == end_id:
                break
                
        if not generated_ids:
            return input_ids

        gen = torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)
        return torch.cat([input_ids, gen], dim=1)

    # --- 省略其他与你原脚本相同的代码 ---
    @property
    def config(self):
        return getattr(self.llm, "config", None)
    
    def gradient_checkpointing_enable(self, *args, **kwargs):
        if self.config is not None:
            try: self.config.use_cache = False
            except Exception: pass
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try: return self.llm.gradient_checkpointing_enable(*args, **kwargs)
            except TypeError: return self.llm.gradient_checkpointing_enable()
        return None
        
    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            try: out = self.llm.gradient_checkpointing_disable()
            except TypeError: out = None
        else: out = None
        if self.config is not None:
            try: self.config.use_cache = True
            except Exception: pass
        return out
        
    @staticmethod
    def _storage_id(t):
        try: return t.untyped_storage().data_ptr()
        except Exception: return t.storage().data_ptr()
        
    def state_dict(self, *args, **kwargs):
        sd = self.llm.state_dict(*args, **kwargs)
        seen = {}
        for k, v in list(sd.items()):
            if not isinstance(v, torch.Tensor): continue
            sid = self._storage_id(v)
            if sid in seen:
                sd[k] = v.clone()
            else:
                seen[sid] = k
        return sd