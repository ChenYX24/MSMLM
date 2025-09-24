# mol_aware_lm_integrated.py
# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
import logging

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .gnn import GVPEncoder
from .mlp import MLPAdapter, DiffusionAdapter
from .tools import extract_and_convert_online
from .lightning_modules_new import LigandOnlyDDPM

# RDKit
from rdkit import Chem

# 日志
logging.getLogger("rdkit").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch.distributed as dist
import os, glob

def has_hf_model_files(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    # 单文件 / 索引文件
    names = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "flax_model.msgpack",
        "tf_model.h5",
    ]
    if any(os.path.isfile(os.path.join(d, n)) for n in names):
        return True
    # 分片文件（无论是否有 index，都当作“该目录包含权重”）
    if glob.glob(os.path.join(d, "model-*-of-*.safetensors")):
        return True
    if glob.glob(os.path.join(d, "pytorch_model-*-of-*.bin")):
        return True
    return False

def any_rank_true(flag: bool) -> bool:
    """只要有一个 rank 为 True，就让所有 rank 都为 True。"""
    if not dist.is_available() or not dist.is_initialized():
        return flag
    t = torch.tensor([1 if flag else 0], device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())

def zero_touch_module(module: torch.nn.Module) -> torch.Tensor:
    """用 0.0 * param.sum() 把 module 接入计算图，不改变 loss 数值。"""
    if module is None:
        return torch.tensor(0.0, device=torch.cuda.current_device())
    z = torch.tensor(0.0, device=next(module.parameters()).device) if any(p.requires_grad for p in module.parameters()) else torch.tensor(0.0, device=torch.cuda.current_device())
    for p in module.parameters():
        if p.requires_grad:
            z = z + (0.0 * p.float().sum())
    return z

def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    简易 position_ids：对每行的有效 token（mask=1）做递增计数，padding 处保持 0。
    与很多 LLM 兼容；若你已有自定义实现，保留你自己的即可。
    """
    # (B, T)
    cumsum = attention_mask.long().cumsum(dim=1) * attention_mask.long()
    # 让从 0 开始：把非零位置减 1
    pos_ids = (cumsum - attention_mask.long()).clamp(min=0)
    return pos_ids

class MolAwareCausalLM(nn.Module):
    """
    集成 NER/GNN/Diffusion 的组合模型；按出现顺序把 <mol> 对应的向量“追加到序列末尾”的虚拟步，
    训练时 labels=-100 不计损，推理时推进 KV 但不出 token。
    """
    # --------------------------- 初始化 ---------------------------
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
        diffusion_config: Optional[Dict] = None,
        diffusion_adapter_config: Optional[Dict] = None,
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

        # ---------- 组件 ----------
        llm_hidden_size = self.llm.config.hidden_size

        # GVPEncoder
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

        # MLP Adapter（把 GVP 向量映射到 LLM 维度）
        mol_adapter_cfg = {
            "input_dim": gvp_encoder_cfg["output_dim"],
            "output_dim": llm_hidden_size,
            "hidden_dim": llm_hidden_size // 2,
            "num_layers": 2,
        }
        if mol_adapter_config:
            mol_adapter_cfg.update(mol_adapter_config)

        # Diffusion 主体（可选）
        self.diffusion = None
        self.diffusion_gen_num_nodes_lig = None  # 可由外部设置
        self.diffusion_cond_dim = diffusion_config.get("cond_dim", llm_hidden_size) if diffusion_config else llm_hidden_size
        if diffusion_config:
            ckpt = diffusion_config.get("checkpoint_path")
            device = diffusion_config.get("device", str(self._first_device()))
            if ckpt and LigandOnlyDDPM is not None:
                try:
                    self.diffusion = LigandOnlyDDPM.load_from_checkpoint(ckpt, map_location=device).to(device)
                    self.diffusion.eval()
                    logging.info(f"Loaded diffusion checkpoint: {ckpt}")
                except Exception as e:
                    logging.warning(f"Failed to load diffusion checkpoint: {e}")

        # Diffusion Adapter
        self.diffusion_adapter = DiffusionAdapter(
            in_dim=llm_hidden_size, out_dim=self.diffusion_cond_dim
        ).to(self._first_device())

        if diffusion_adapter_config:
            ckpt = diffusion_adapter_config.get("ckpt_path")
            if ckpt and os.path.isfile(ckpt):
                try:
                    sd = torch.load(ckpt, map_location=self._first_device())
                    self.diffusion_adapter.load_state_dict(sd, strict=True)
                    logging.info(f"Loaded diffusion_adapter weights from: {ckpt}")
                except Exception as e:
                    logging.warning(f"Failed to load diffusion_adapter weights: {e}")

        self.gvp_encoder = GVPEncoder(**gvp_encoder_cfg).to(self._first_device())
        self.mol_adapter = MLPAdapter(**mol_adapter_cfg).to(self._first_device())
        self.smiles_cache: Dict[str, str] = {}

        # ---------- 关键：HF Trainer 兼容字段 ----------
        # 让 Trainer 把它当作 PreTrainedModel 一样保存
        self._config = getattr(self.llm, "config", None)
        self._keys_to_ignore_on_save = getattr(self.llm, "_keys_to_ignore_on_save", None)
        self._keys_to_ignore_on_load_missing = getattr(self.llm, "_keys_to_ignore_on_load_missing", None)
        self._keys_to_ignore_on_load_unexpected = getattr(self.llm, "_keys_to_ignore_on_load_unexpected", None)

    # --------------------------- HF 兼容接口 ---------------------------
    @property
    def config(self):
        return self._config

    @property
    def _keys_to_ignore_on_save(self):
        return getattr(self.llm, "_keys_to_ignore_on_save", [])

    @_keys_to_ignore_on_save.setter
    def _keys_to_ignore_on_save(self, v):
        # 仅为了避免 AttributeError；Trainer 不需要我们真正改 llm 的字段
        self.__dict__["__keys_to_ignore_on_save"] = v

    @property
    def _keys_to_ignore_on_load_missing(self):
        return getattr(self.llm, "_keys_to_ignore_on_load_missing", [])

    @_keys_to_ignore_on_load_missing.setter
    def _keys_to_ignore_on_load_missing(self, v):
        self.__dict__["__keys_to_ignore_on_load_missing"] = v

    @property
    def _keys_to_ignore_on_load_unexpected(self):
        return getattr(self.llm, "_keys_to_ignore_on_load_unexpected", [])

    @_keys_to_ignore_on_load_unexpected.setter
    def _keys_to_ignore_on_load_unexpected(self, v):
        self.__dict__["__keys_to_ignore_on_load_unexpected"] = v

    def to(self, *args, **kwargs):
        # 同步把底座 LLM 与自定义模块都迁移设备
        super().to(*args, **kwargs)
        self.llm.to(*args, **kwargs)
        self.gvp_encoder.to(*args, **kwargs)
        self.mol_adapter.to(*args, **kwargs)
        self.diffusion_adapter.to(*args, **kwargs)
        if self.diffusion is not None:
            try:
                self.diffusion.to(*args, **kwargs)
            except Exception:
                pass
        return self

    # --------------------------- 辅助 ---------------------------
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

    def _get_last_hidden_before_pos(self, row_ids: torch.Tensor, end_pos: int) -> torch.Tensor:
        assert end_pos > 0, "end_pos should be > 0"
        dev = self._first_device()
        prefix = row_ids[:end_pos].unsqueeze(0).to(dev)
        attn = (prefix != self.pad_token_id).long().to(dev)
        out = self.llm(input_ids=prefix, attention_mask=attn,
                       output_hidden_states=True, use_cache=False, return_dict=True)
        return out.hidden_states[-1][0, -1, :].detach()

    def _black_box_from_hidden_hctx(self, h_ctx: torch.Tensor) -> Optional[torch.Tensor]:
        if self.diffusion is None or LigandOnlyDDPM is None or Chem is None:
            logging.warning("Diffusion not available; skip virtual step.")
            return None
        dev = self._first_device()
        cond = self.diffusion_adapter(h_ctx.to(dev)).float().unsqueeze(0)  # [1, cond_dim]
        try:
            gen_mols = self.diffusion.generate_mol_from_embedding(
                batch_size=1, embeddings=cond, num_nodes_lig=self.diffusion_gen_num_nodes_lig
            )
            gen_mol = gen_mols[0] if isinstance(gen_mols, (list, tuple)) else gen_mols
            gen_smiles = Chem.MolToSmiles(gen_mol) if gen_mol is not None else None
        except Exception as e:
            logging.warning(f"Diffusion generation failed: {e}")
            gen_smiles = None
        if not gen_smiles:
            return None
        gvp_embedding = self.gvp_encoder.forward_from_smiles(gen_smiles).squeeze(0)
        # todo(如果报错删这一行和下一行)
        logging.info("🔁 (Offline) Got SMILES from diffusion; using GVP.")
        return self.mol_adapter(gvp_embedding)

    def _black_box_embed_offline(
        self,
        row_ids: torch.Tensor,
        row_embeds: torch.Tensor,
        row_mask: torch.Tensor,
        pos_mol: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        # 1) 尝试从上下文抽取 SMILES
        llm_context = self.tokenizer.decode(row_ids[:pos_mol].tolist(), skip_special_tokens=True)
        smiles = self._get_smiles_from_context(llm_context)
        if smiles:
            gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
            # todo(如果报错删这一行和下一行)
            logging.info("✅ (Offline) Found SMILES; using GVP.")
            return self.mol_adapter(gvp_embedding)
        # 2) 兜底：用 h_ctx 走 diffusion -> SMILES -> GVP
        if self.diffusion is None or LigandOnlyDDPM is None or Chem is None:
            logging.warning("❌ (Offline) Diffusion not available; skip virtual step.")
            return None
        h_ctx = self._get_last_hidden_before_pos(row_ids, pos_mol)  # [H]
        return self._black_box_from_hidden_hctx(h_ctx)

    def _black_box_embed_online(
        self,
        llm_context_text: Optional[str] = None,
        context_ids: Optional[torch.Tensor] = None,
        h_ctx: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if llm_context_text:
            smiles = self._get_smiles_from_context(llm_context_text)
            if smiles:
                gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
                return self.mol_adapter(gvp_embedding)
        if h_ctx is not None:
            return self._black_box_from_hidden_hctx(h_ctx)
        if context_ids is None and llm_context_text is not None:
            dev = self._first_device()
            toks = self.tokenizer(llm_context_text, return_tensors="pt", add_special_tokens=False)
            context_ids = toks["input_ids"].to(dev)
        if context_ids is not None:
            attn = (context_ids != self.pad_token_id).long().to(context_ids.device)
            out = self.llm(
                input_ids=context_ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False, return_dict=True
            )
            h_ctx = out.hidden_states[-1][0, -1, :].detach()
            return self._black_box_from_hidden_hctx(h_ctx)
        return None

    # --------------------------- 训练/评估前向 ---------------------------
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> CausalLMOutputWithPast:
            assert input_ids is not None, "MolAwareCausalLM 需要 input_ids"

            # 1) 先离线拼接 <mol> 的嵌入到序列末尾（仅在存在 <mol> 时追加）
            new_embeds, new_masks, new_labels, appended_mol_cnt = self._append_mol_embeds_to_end_offline(
                input_ids, attention_mask, labels
            )

            # 2) 常规 LLM 前向
            position_ids = build_position_ids(new_masks).to(new_masks.device)
            outputs = self.llm(
                inputs_embeds=new_embeds,
                attention_mask=new_masks,
                position_ids=position_ids,
                labels=new_labels,
                return_dict=True,
                **kwargs,
            )

            # 3) —— DDP 安全处理 ——：
            # “本 rank 是否真的追加过 mol 向量”
            used_mol_local = (appended_mol_cnt > 0)
            # “所有 rank 是否至少有一个用到 mol 分支”
            used_mol_global = any_rank_true(used_mol_local)

            if used_mol_global and (not used_mol_local) and (outputs.loss is not None):
                if hasattr(self, "mol_adapter"):
                    outputs.loss = outputs.loss + zero_touch_module(self.mol_adapter)
                if hasattr(self, "gnn_mlp"):
                    outputs.loss = outputs.loss + zero_touch_module(self.gnn_mlp)
                if hasattr(self, "diffusion_mlp"):
                    outputs.loss = outputs.loss + zero_touch_module(self.diffusion_mlp)

            return outputs


    def _append_mol_embeds_to_end_offline(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
            """
            将 batch 内每个样本中出现 <mol> 的位置，调用黑盒/外部编码器得到的分子向量，
            直接“追加”到该序列末尾；相应的 mask/labels 做齐。
            返回：new_embeds, new_masks, new_labels, appended_mol_cnt_total
            其中 appended_mol_cnt_total 统计本 rank 本步实际追加 mol 向量的“个数”（用于 DDP 同步判断）。
            """
            assert input_ids.dim() == 2, "input_ids 形状应为 (B, T)"
            embed_tokens = self.llm.get_input_embeddings()
            emb_dev = embed_tokens.weight.device

            input_ids = input_ids.to(emb_dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(emb_dev)
            if labels is not None:
                labels = labels.to(emb_dev)

            B, T = input_ids.shape
            device = input_ids.device
            embeds = embed_tokens(input_ids)         # (B, T, D)
            D = embeds.size(-1)

            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long().to(device)
            has_labels = labels is not None

            rows_embeds, rows_masks, rows_labels = [], [], []
            max_len = 0
            appended_mol_cnt_total = 0  # <<< 关键：累计追加的 mol 向量数（仅在真正追加时 +1）

            for b in range(B):
                row_ids = input_ids[b]          # (T,)
                row_emb = embeds[b]             # (T, D)
                row_msk = attention_mask[b]     # (T,)
                row_lbl = labels[b] if has_labels else None

                # 先把原始 token 的 embed/mask/label 按顺序压入
                new_emb_list = [row_emb[i] for i in range(T)]
                new_msk_list = [int(row_msk[i].item()) for i in range(T)]
                new_lbl_list = [int(row_lbl[i].item()) for i in range(T)] if has_labels else None

                # 找到 <mol> 的位置
                mol_positions = (row_ids == self.mol_token_id).nonzero(as_tuple=False).flatten().tolist()
                for p in mol_positions:
                    # 若该位置是 padding（mask=0），跳过
                    if new_msk_list[p] == 0:
                        continue

                    mol_emb = self._black_box_embed_offline(row_ids, row_emb, row_msk, p)
                    if mol_emb is None:
                        if getattr(self, "debug", False):
                            import logging
                            logging.info("[Offline] Skip virtual step for <mol> (no embedding).")
                        continue

                    # 真实追加
                    new_emb_list.append(mol_emb)   # 末尾再加一个 token 向量
                    new_msk_list.append(1)         # 有效
                    if has_labels:
                        new_lbl_list.append(-100)  # 不参与 loss
                    appended_mol_cnt_total += 1

                # 本样本的拼接结果 -> tensor
                new_len = len(new_msk_list)
                max_len = max(max_len, new_len)

                new_emb = torch.stack(new_emb_list, dim=0)                             # (L, D)
                new_msk = torch.tensor(new_msk_list, device=device, dtype=row_msk.dtype)  # (L,)
                new_lbl = (torch.tensor(new_lbl_list, device=device, dtype=input_ids.dtype)
                        if has_labels else None)

                rows_embeds.append(new_emb)
                rows_masks.append(new_msk)
                if has_labels:
                    rows_labels.append(new_lbl)

            # 对齐到同一长度（右侧 padding）
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

                padded_embeds.append(E.unsqueeze(0))  # (1, max_len, D)
                padded_masks.append(M.unsqueeze(0))   # (1, max_len)
                if has_labels:
                    padded_labels.append(L.unsqueeze(0) if L is not None else None)

            new_embeds = torch.cat(padded_embeds, dim=0)              # (B, max_len, D)
            new_masks  = torch.cat(padded_masks,  dim=0)              # (B, max_len)
            new_labels = torch.cat(padded_labels, dim=0) if has_labels else None  # (B, max_len) or None

            if getattr(self, "debug", False):
                orig_tokens = attention_mask.sum().item()
                new_tokens  = new_masks.sum().item()
                print(f"[MolAware/offline] appended {int(new_tokens - orig_tokens)} embeddings to batch end; "
                    f"mol_appended_count={appended_mol_cnt_total}")

            return new_embeds, new_masks, new_labels, appended_mol_cnt_total

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
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        past = outputs.past_key_values
        attn_mask = attention_mask
        generated_ids: List[int] = []
        end_id = self.eos_token_id if eos_token_id is None else eos_token_id

        def _apply_sampling(logits: torch.Tensor) -> torch.Tensor:
            if do_sample:
                _logits = logits
                if temperature and temperature != 1.0:
                    _logits = _logits / temperature
                if top_k and top_k > 0:
                    v, _ = torch.topk(_logits, top_k)
                    _logits = _logits.masked_fill(_logits < v[:, [-1]], float("-inf"))
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(_logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = probs.cumsum(dim=-1)
                    cutoff = (cumprobs > top_p).float().cumsum(dim=-1).bool()
                    sorted_logits[cutoff] = float("-inf")
                    _logits = torch.full_like(_logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
                return torch.multinomial(torch.softmax(_logits, dim=-1), num_samples=1)
            else:
                return torch.argmax(logits, dim=-1, keepdim=True)

        for _ in range(max_new_tokens):
            logits = outputs.logits[:, -1, :]

            # 简单版重复惩罚
            if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
                uniq = list(set(generated_ids))
                logits[:, uniq] = logits[:, uniq] / repetition_penalty

            last_hidden = outputs.hidden_states[-1][:, -1, :].detach()  # [1, H]
            h_ctx_step = last_hidden[0]

            next_token = _apply_sampling(logits)
            next_id = int(next_token.item())

            if next_id == self.mol_token_id:
                current_context_ids = torch.cat(
                    [input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)],
                    dim=1
                )
                llm_context_text = self.tokenizer.decode(
                    current_context_ids[0].tolist(), skip_special_tokens=True
                )

                mol_embedding = self._black_box_embed_online(
                    llm_context_text=llm_context_text,
                    context_ids=None,
                    h_ctx=h_ctx_step,
                )

                if mol_embedding is None:
                    logits_block = logits.clone()
                    logits_block[0, self.mol_token_id] = float("-inf")
                    next_token = _apply_sampling(logits_block)
                    next_id = int(next_token.item())
                else:
                    outputs = llm(
                        inputs_embeds=mol_embedding.view(1, 1, -1),
                        attention_mask=torch.cat(
                            [attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1
                        ),
                        past_key_values=past,
                        use_cache=use_cache,
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs,
                    )
                    past = outputs.past_key_values
                    attn_mask = torch.cat(
                        [attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1
                    )
                    continue

            step_ids = next_token  # [1,1]
            attn_mask = torch.cat(
                [attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1
            )
            outputs = llm(
                input_ids=step_ids,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=use_cache,
                output_hidden_states=True,
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

    # --------------------------- HF 保存/加载 ---------------------------
    def state_dict(self, *args, **kwargs):
        # 保存整个组合模型的权重（包含自定义模块 + 底座 llm 的参数拷贝）
        sd = super().state_dict(*args, **kwargs)
        # 去重相同 storage，避免稀奇的共享 tensor 被重复引用
        seen = {}
        for k, v in list(sd.items()):
            if not isinstance(v, torch.Tensor):
                continue
            sid = self._storage_id(v)
            if sid in seen:
                sd[k] = v.clone()
            else:
                seen[sid] = k
        return sd

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        - 先调用底座 LLM 的 save_pretrained（保存权重、config 等）
        - 再额外保存组合模型的自定义模块（.pt）
        - 写入一个 metadata.json 记录额外文件名，便于 from_pretrained 恢复
        """
        os.makedirs(save_directory, exist_ok=True)
        # 1) 保存底座 LLM
        out = self.llm.save_pretrained(save_directory, **kwargs)

        # 2) 额外保存自定义模块
        extras = {}
        if hasattr(self, "gvp_encoder") and self.gvp_encoder is not None:
            torch.save(self.gvp_encoder.state_dict(), os.path.join(save_directory, "gvp_encoder.pt"))
            extras["gvp_encoder"] = "gvp_encoder.pt"
        if hasattr(self, "mol_adapter") and self.mol_adapter is not None:
            torch.save(self.mol_adapter.state_dict(), os.path.join(save_directory, "mol_adapter.pt"))
            extras["mol_adapter"] = "mol_adapter.pt"
        if hasattr(self, "diffusion_adapter") and self.diffusion_adapter is not None:
            torch.save(self.diffusion_adapter.state_dict(), os.path.join(save_directory, "diffusion_adapter.pt"))
            extras["diffusion_adapter"] = "diffusion_adapter.pt"

        # diffusion 主体通常体量较大且可选，不强制保存；如果需要自行加：
        # if hasattr(self, "diffusion") and self.diffusion is not None:
        #     torch.save(self.diffusion.state_dict(), os.path.join(save_directory, "diffusion.pt"))
        #     extras["diffusion"] = "diffusion.pt"

        meta = {
            "class": "MolAwareCausalLM",
            "version": 1,
            "extras": extras,
            "mol_token": self.mol_token,
        }
        with open(os.path.join(save_directory, "molaware_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return out

    @classmethod
    def from_pretrained(cls, save_directory: str, tokenizer=None,
                        diffusion_config=None, diffusion_adapter_config=None,
                        **kwargs):
        root = save_directory
        meta_path = os.path.join(root, "molaware_metadata.json")
        has_meta = os.path.isfile(meta_path)

        # 1) 解析 metadata（若存在）
        meta = {}
        extras_map = {}
        if has_meta:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            extras_map = meta.get("extras", {}) or {}

        # 2) 决定 LLM 目录：优先 <root>/llm，其次 <root>
        llm_dir = os.path.join(root, "llm")
        if not has_hf_model_files(llm_dir):
            llm_dir = root
        print(f"[from_pretrained] using llm_dir={llm_dir}")

        # 3) 加载底座 LLM
        base_llm = AutoModelForCausalLM.from_pretrained(llm_dir, **kwargs)

        # 4) tokenizer：若未传入，则用根目录（因为 tokenizer 保存在根）
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)

        # 5) 构造实例
        model = cls(llm=base_llm, tokenizer=tokenizer,
                    diffusion_config=diffusion_config,
                    diffusion_adapter_config=diffusion_adapter_config)

        # 6) 加载 extras（按 metadata 的相对路径）
        def _maybe_load_sub(sd_path, module_attr):
            if not sd_path:
                return
            path = os.path.join(root, sd_path) if not os.path.isabs(sd_path) else sd_path
            if os.path.isfile(path):
                sd = torch.load(path, map_location="cpu")
                mod = getattr(model, module_attr, None)
                if mod is not None and hasattr(mod, "load_state_dict"):
                    # 兼容直接存 state_dict（keys 裸的）或者带前缀；使用 strict=False 更韧性
                    mod.load_state_dict(sd, strict=False)

        if has_meta:
            _maybe_load_sub(extras_map.get("gvp_encoder"), "gvp_encoder")
            _maybe_load_sub(extras_map.get("mol_adapter"), "mol_adapter")
            _maybe_load_sub(extras_map.get("diffusion_adapter"), "diffusion_adapter")
            _maybe_load_sub(extras_map.get("diffusion"), "diffusion")  # 只有你真的保存了 diffusion.pt 才会有

        return model


    # --------------------------- 其它辅助 ---------------------------
    def gradient_checkpointing_enable(self, *args, **kwargs):
        if self.config is not None:
            try:
                self.config.use_cache = False
            except Exception:
                pass
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                return self.llm.gradient_checkpointing_enable(*args, **kwargs)
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
    def _storage_id(t: torch.Tensor):
        try:
            return t.untyped_storage().data_ptr()
        except Exception:
            return t.storage().data_ptr()
