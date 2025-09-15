# mol_aware_lm_integrated.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
import logging
import os

# 导入你提供的原始模块
from .gnn import GVPEncoder
from .mlp import MLPAdapter, DiffusionAdapter
from .tools import extract_and_convert_online
from transformers.modeling_outputs import CausalLMOutputWithPast

from .lightning_modules_new import LigandOnlyDDPM
from rdkit import Chem


# 禁用RDKit日志
logging.getLogger("rdkit").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    cumsum = attention_mask.long().cumsum(dim=-1)
    pos = (cumsum - 1).clamp(min=0)
    return pos * attention_mask.long()

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
        
        # --- 1. init: 添加 gvp_encoder, mol_adapter 和 diffusion ---
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
        
        # ---------- Diffusion encoder (DDPM) ----------
        self.diffusion = None
        self.diffusion_cond_dim = llm_hidden_size  # 默认同 hidden_size
        if diffusion_config:
            self.diffusion_cond_dim = diffusion_config.get("cond_dim", llm_hidden_size)
            ckpt = diffusion_config.get("checkpoint_path")
            device = diffusion_config.get("device", str(self._first_device()))
            if ckpt and LigandOnlyDDPM is not None:
                try:
                    self.diffusion = LigandOnlyDDPM.load_from_checkpoint(ckpt, map_location=device).to(device)
                    self.diffusion.eval()
                    logging.info(f"Loaded diffusion checkpoint: {ckpt}")
                except Exception as e:
                    logging.warning(f"Failed to load diffusion checkpoint: {e}")

        # ---------- Diffusion adapter (= diffusion_mlp) ----------
        # 统一命名：你的训练里的 MLP 就是我们这里的 diffusion_adapter
        self.diffusion_adapter = DiffusionAdapter(
            in_dim=llm_hidden_size,
            out_dim=self.diffusion_cond_dim,
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
        """
        已有 h_ctx（<mol> 之前最后一个 token 的最后一层隐状态，shape=[H]）。
        走：h_ctx -> diffusion_adapter -> diffusion.generate -> RDKit 解析 -> GVP -> mol_adapter
        解析失败则返回 None（不插虚拟步）。
        """
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
        return self.mol_adapter(gvp_embedding)

    def _black_box_embed_offline(
        self,
        row_ids: torch.Tensor,
        row_embeds: torch.Tensor,
        row_mask: torch.Tensor,
        pos_mol: int,
    ) -> Optional[torch.Tensor]:
        """
        优先：从上下文识别 SMILES -> GVP -> mol_adapter。
        否则：h_ctx -> diffusion_adapter -> diffusion.generate -> 解析 SMILES -> 若成功再走 GVP；否则返回 None（不插入虚拟步）。
        """
        llm_context = self.tokenizer.decode(row_ids[:pos_mol].tolist(), skip_special_tokens=True)
        smiles = self._get_smiles_from_context(llm_context)

        if smiles:
            logging.info("✅ (Offline) Found SMILES; using GVP.")
            gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
            return self.mol_adapter(gvp_embedding)

        # 无 SMILES：尝试扩散生成 → 解析 → 再走 GVP
        if self.diffusion is None or LigandOnlyDDPM is None or Chem is None:
            logging.warning("❌ (Offline) Diffusion not available; skip virtual step.")
            return None

        h_ctx = self._get_last_hidden_before_pos(row_ids, pos_mol)  # [H]
        cond = self.diffusion_adapter(h_ctx).float().unsqueeze(0)   # [1, cond_dim]

        try:
            gen_mols = self.diffusion.generate_mol_from_embedding(batch_size=1, embeddings=cond, num_nodes_lig=None)
            gen_mol = gen_mols[0] if isinstance(gen_mols, (list, tuple)) else gen_mols
            gen_smiles = Chem.MolToSmiles(gen_mol) if gen_mol is not None else None
        except Exception as e:
            logging.warning(f"Diffusion generation failed: {e}")
            gen_smiles = None

        if gen_smiles:
            logging.info("🔁 (Offline) Got SMILES from diffusion; using GVP.")
            gvp_embedding = self.gvp_encoder.forward_from_smiles(gen_smiles).squeeze(0)
            return self.mol_adapter(gvp_embedding)

        logging.warning("❌ (Offline) No SMILES after diffusion; skip virtual step.")
        return None


    def _black_box_embed_online(
        self,
        llm_context_text: str,
        context_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        推理路径：同 offline 逻辑；拿不到 SMILES 时返回 None，调用方据此不插虚拟步。
        """
        smiles = self._get_smiles_from_context(llm_context_text)
        if smiles:
            logging.info("✅ (Online) Found SMILES; using GVP.")
            gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
            return self.mol_adapter(gvp_embedding)

        if self.diffusion is None or LigandOnlyDDPM is None or Chem is None:
            logging.warning("❌ (Online) Diffusion not available; skip virtual step.")
            return None

        dev = self._first_device()
        if context_ids is None:
            toks = self.tokenizer(llm_context_text, return_tensors="pt", add_special_tokens=False)
            context_ids = toks["input_ids"].to(dev)
        attn = (context_ids != self.pad_token_id).long().to(dev)
        out = self.llm(input_ids=context_ids, attention_mask=attn,
                       output_hidden_states=True, use_cache=False, return_dict=True)
        h_ctx = out.hidden_states[-1][0, -1, :].detach()          # [H]
        cond = self.diffusion_adapter(h_ctx).float().unsqueeze(0) # [1, cond_dim]

        try:
            gen_mols = self.diffusion.generate_mol_from_embedding(batch_size=1, embeddings=cond, num_nodes_lig=None)
            gen_mol = gen_mols[0] if isinstance(gen_mols, (list, tuple)) else gen_mols
            gen_smiles = Chem.MolToSmiles(gen_mol) if gen_mol is not None else None
        except Exception as e:
            logging.warning(f"Diffusion generation failed: {e}")
            gen_smiles = None

        if gen_smiles:
            logging.info("🔁 (Online) Got SMILES from diffusion; using GVP.")
            gvp_embedding = self.gvp_encoder.forward_from_smiles(gen_smiles).squeeze(0)
            return self.mol_adapter(gvp_embedding)

        logging.warning("❌ (Online) No SMILES after diffusion; skip virtual step.")
        return None

    def _black_box_embed_online(
        self,
        llm_context_text: Optional[str] = None,
        context_ids: Optional[torch.Tensor] = None,
        h_ctx: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        优先流程：
        1) 先从文本里解析 SMILES → GVP → mol_adapter
        2) 若无 SMILES：
            - 若提供了 h_ctx：直接 _black_box_from_hidden_hctx(h_ctx)
            - 否则再跑一次 prefix 前向拿 h_ctx（保留旧逻辑做兜底）
        """
        # 先用上下文直接识别 SMILES（如果给了）
        if llm_context_text:
            smiles = self._get_smiles_from_context(llm_context_text)
            if smiles:
                gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
                return self.mol_adapter(gvp_embedding)

        # 没识别到 SMILES：如果有 h_ctx，直接用
        if h_ctx is not None:
            return self._black_box_from_hidden_hctx(h_ctx)

        # 兜底：没有 h_ctx，就做一次前向拿 h_ctx（和原实现一致）
        if context_ids is None and llm_context_text is not None:
            dev = self._first_device()
            toks = self.tokenizer(llm_context_text, return_tensors="pt", add_special_tokens=False)
            context_ids = toks["input_ids"].to(dev)
        if context_ids is not None:
            attn = (context_ids != self.pad_token_id).long().to(context_ids.device)
            out = self.llm(
                input_ids=context_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            h_ctx = out.hidden_states[-1][0, -1, :].detach()
            return self._black_box_from_hidden_hctx(h_ctx)

        return None


    # ---------- 前向（训练/评估） ----------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs, 
    ) -> CausalLMOutputWithPast:
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
                mol_emb = self._black_box_embed_offline(row_ids, row_emb, row_msk, p)
                if mol_emb is None:
                    # 不追加虚拟步，保持原序列不变
                    if self.debug:
                        logging.info("[Offline] Skip virtual step for <mol> (no SMILES).")
                    continue
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
        """
        实时推理：遇到 <mol> 时优先用 llm_context_text 解析 SMILES；
        若无则用当前步 h_ctx 走 diffusion→解析；失败则屏蔽 <mol> 重采样。
        """
        use_cache = kwargs.pop("use_cache", True)
        try:
            self.llm.config.use_cache = True
        except Exception:
            pass

        if not realtime_mol:
            # 非实时路径，原样透传
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

        # 首次前向
        outputs = llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,   # 关键：拿 h_ctx
            return_dict=True,
            **kwargs,
        )
        past = outputs.past_key_values
        attn_mask = attention_mask
        generated_ids: List[int] = []
        end_id = self.eos_token_id if eos_token_id is None else eos_token_id

        def _apply_sampling(logits: torch.Tensor) -> torch.Tensor:
            # 按当前设置做一次采样/贪心
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

            # Repetition penalty（简单版）
            if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
                uniq = list(set(generated_ids))
                logits[:, uniq] = logits[:, uniq] / repetition_penalty

            # 取本步 h_ctx（最后一层、最后一个 token）
            last_hidden = outputs.hidden_states[-1][:, -1, :].detach()  # [1, H]
            h_ctx_step = last_hidden[0]                                  # [H]

            # 先出一个候选 token
            next_token = _apply_sampling(logits)
            next_id = int(next_token.item())

            if next_id == self.mol_token_id:
                # 构造 llm_context_text（不含将要输出的 token）
                current_context_ids = torch.cat(
                    [input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)],
                    dim=1
                )
                llm_context_text = self.tokenizer.decode(
                    current_context_ids[0].tolist(), skip_special_tokens=True
                )

                # 优先：用文本解析 SMILES；否则用 h_ctx 走 diffusion→解析
                mol_embedding = self._black_box_embed_online(
                    llm_context_text=llm_context_text,
                    context_ids=None,
                    h_ctx=h_ctx_step,
                )

                if mol_embedding is None:
                    # 不插虚拟步：屏蔽 <mol> 并重采样
                    logits_block = logits.clone()
                    logits_block[0, self.mol_token_id] = float("-inf")
                    next_token = _apply_sampling(logits_block)
                    next_id = int(next_token.item())
                else:
                    # 插入虚拟步：推进 KV / attn，不计入输出
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
                    # 继续下一轮预测（不把 <mol> 计入输出）
                    continue

            # 正常 token 前进一步：推进 KV、记录输出
            step_ids = next_token  # [1, 1]
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