import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import wandb
import os
import torch.nn.functional as F
import numpy as np
from lightning_modules_new import LigandOnlyDDPM
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# ========== Utilities ==========

def compute_rmsd(coords1, coords2):
    coords1 = np.array(coords1)
    coords2 = np.array(coords2[:coords1.shape[0]])
    if coords1.shape != coords2.shape:
        return None
    diff = coords1 - coords2
    return float(np.sqrt((diff ** 2).sum() / len(coords1)))

def get_coords_and_atoms_without_R(mol):
    if mol is None:
        return [], []
    keep = [(atom.GetIdx(), atom.GetSymbol()) for atom in mol.GetAtoms()]
    if not keep:
        return [], []
    conf = mol.GetConformer()
    coords = [list(conf.GetAtomPosition(i)) for i, _ in keep]
    atoms = [symbol for _, symbol in keep]
    return coords, atoms

def sort_coords_and_atoms_by_encoder(coords, atoms, atom_encoder):
    atom_order = {k: v for k, v in atom_encoder.items()}
    fallback_idx = atom_order.get("others", len(atom_order))
    pairs = list(zip(coords, atoms))
    sorted_pairs = sorted(pairs, key=lambda x: atom_order.get(x[1], fallback_idx))
    sorted_coords = [p[0] for p in sorted_pairs]
    sorted_atoms = [p[1] for p in sorted_pairs]
    return sorted_coords, sorted_atoms

def save_mols_as_sdf(gen_mols, names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for mol, name in zip(gen_mols, names):
        if mol is None:
            continue
        file_path = os.path.join(output_dir, f"{name}.sdf")
        if not mol.GetNumConformers():
            AllChem.EmbedMolecule(mol)
        writer = Chem.SDWriter(file_path)
        writer.write(mol)
        writer.close()

# ========== Constants ==========

atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10}
num_types = len(atom_encoder)

# ========== Dataset ==========

class MolDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ========== Models ==========

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),   
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)    
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(x)


def extract_mol_span_emb(input_ids, hidden_states, tokenizer, mol_token="<mol>"):
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)
    if mol_token not in token_strs:
        raise ValueError(f"{mol_token} not found in {token_strs}")
    mol_idx = token_strs.index(mol_token)
    span_emb = hidden_states[0, :mol_idx, :]  # [L, H]
    emb0 = span_emb[-1]
    return emb0

class GNNModel(nn.Module):
    pass

class MolTrainer(nn.Module):
    def __init__(self, llm_path, diffusion_ckpt_path, gnn, train_mode, cond_dim=3072, device='cuda'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.float16
        ).to("cuda:2")
        self.llm2 = AutoModelForCausalLM.from_pretrained(
            "/data1/chenyuxuan/Project/MSMLM/model/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95", torch_dtype=torch.float16
        ).to("cuda:3")
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.diffusion = LigandOnlyDDPM.load_from_checkpoint(diffusion_ckpt_path, map_location=device).to('cuda:0')
        self.diffusion.eval()

        self.device = device
        self.atom_encoder = atom_encoder
        self.atom_decoder = {v: k for k, v in atom_encoder.items()}

        llm_hidden_size = self.llm.config.hidden_size
        diffusion_cond_dim = cond_dim

        self.diffusion_mlp = MLP(llm_hidden_size, diffusion_cond_dim).to(device)
        # self.gnn = gnn.to(device)
        # gnn_hidden_size = self.gnn.emb_dim
        # self.gnn_mlp = MLP(gnn_hidden_size, llm_hidden_size).to(device)

        self.train_mode = train_mode
        # self.freeze_params()

    def freeze_params(self):
        if self.train_mode == 'gnn':
            for p in self.diffusion_mlp.parameters():
                p.requires_grad = False
        elif self.train_mode == 'diffusion':
            for p in self.gnn.parameters():
                p.requires_grad = False
            for p in self.gnn_mlp.parameters():
                p.requires_grad = False

    def count_f1(self, pred_counter: torch.Tensor, gt_counter: torch.Tensor):
        # 保证两个 counter 长度一致
        assert pred_counter.shape == gt_counter.shape, "Counters must have the same shape"

        # True Positive: 预测正确的数量（即 min(pred, gt)）
        tp = torch.minimum(pred_counter, gt_counter).sum().float()

        # False Positive: 预测多了（预测有但GT没有）
        fp = torch.clamp(pred_counter - gt_counter, min=0).sum().float()

        # False Negative: 预测少了（GT有但预测没有）
        fn = torch.clamp(gt_counter - pred_counter, min=0).sum().float()

        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision.item(), recall.item(), f1.item()
    
    def forward(self, batch):
        input_texts = batch["input_text"]
        smiles = batch["smiles"]
        molecules = batch["molecule"]
        orig_emb1 = batch.get("orig_emb1", None)
        if orig_emb1 == None:
            tokens = self.tokenizer(input_texts, return_tensors="pt", padding=True).to(self.llm2.device)
            outputs = self.llm2(**tokens, output_hidden_states=True, return_dict=True)
            orig_emb1 = torch.stack([
                extract_mol_span_emb(tokens['input_ids'][i], outputs.hidden_states[-1][i:i+1], self.tokenizer)
                for i in range(len(input_texts))
            ])
        tokens = self.tokenizer(input_texts, return_tensors="pt", padding=True).to(self.llm.device)
        outputs = self.llm(**tokens, output_hidden_states=True, return_dict=True)
        emb0 = torch.stack([
            extract_mol_span_emb(tokens['input_ids'][i], outputs.hidden_states[-1][i:i+1], self.tokenizer)
            for i in range(len(input_texts))
        ])
        emb0 = emb0.to(dtype=torch.float32, device=self.device)
        orig_emb1 = orig_emb1.to(dtype=torch.float32, device=self.device)
        emb1 = self.diffusion_mlp(emb0)
        diffusion_emb_loss = nn.MSELoss()(emb1, orig_emb1)
        
        with torch.no_grad():
            gen_mols = self.diffusion.generate_mol_from_embedding(
                batch_size=len(emb1), embeddings=emb1, num_nodes_lig=None
            )

        coord_losses, type_recalls = [], []
        for i in range(len(gen_mols)):
            pred_coords, pred_atoms = get_coords_and_atoms_without_R(gen_mols[i])
            gt_coords = molecules[i]["coordinates"]
            gt_atoms = molecules[i]["atom_types"]

            pred_coords, pred_atoms = sort_coords_and_atoms_by_encoder(pred_coords, pred_atoms, self.atom_encoder)
            gt_coords, gt_atoms = sort_coords_and_atoms_by_encoder(gt_coords, gt_atoms, self.atom_encoder)

            if len(gt_coords) == 0 or len(pred_coords) == 0:
                coord_losses.append(torch.tensor(10.0, device=self.device))
                type_recalls.append(torch.tensor(0.0, device=self.device))
                continue

            rmsd = compute_rmsd(gt_coords, pred_coords) if len(gt_coords) <= len(pred_coords) else compute_rmsd(pred_coords, gt_coords)
            coord_losses.append(torch.tensor(rmsd if rmsd is not None else 10.0, device=self.device))

            pred_ids = [self.atom_encoder.get(a, self.atom_encoder['others']) for a in pred_atoms]
            gt_ids = [self.atom_encoder.get(a, self.atom_encoder['others']) for a in gt_atoms]
            pred_counter = torch.bincount(torch.tensor(pred_ids, device=self.device), minlength=num_types)
            gt_counter = torch.bincount(torch.tensor(gt_ids, device=self.device), minlength=num_types)
            precision, recall, f1 = self.count_f1(pred_counter, gt_counter)
            type_recalls.append(torch.tensor(f1, device=self.device))

        coord_loss = torch.stack(coord_losses).mean()
        type_loss = torch.stack(type_recalls).mean()
        diffusion_total_loss = coord_loss + type_loss

        # gnn_emb = self.gnn(smiles)
        # emb2 = self.gnn_mlp(gnn_emb)
        gnn_align_loss = 0
        # gnn_align_loss = nn.MSELoss()(emb2, emb0)

        total_loss = {
            'gnn': gnn_align_loss,
            'diffusion': diffusion_emb_loss,
            'both': diffusion_emb_loss + gnn_align_loss
        }[self.train_mode]

        return diffusion_emb_loss, diffusion_total_loss, coord_loss, type_loss, gnn_align_loss, total_loss

# ========== Collate ==========

# def collate_fn(batch):        
#     return {
#         "input_text": [x["input_text"] for x in batch],
#         "smiles": [x["smiles"] for x in batch],
#         "molecule": [x["molecule"] for x in batch],
#     }

def collate_fn(batch):
    input_texts = []
    smiles_list = []
    molecules = []
    orig_emb1_list = []
    for x in batch:
        smiles = x['smiles']
        input_texts.append(smiles + '<mol>')
        smiles_list.append(smiles)
        orig_emb1_list.append(torch.tensor(x['embedding'], dtype=torch.float16))
        molecule = {
            "coordinates": x["coordinates"],  # 通常为 list of [x, y, z]
            "atom_types": x["atom_types"]     # list of atom symbols or atom type ints
        }
        molecules.append(molecule)

    return {
        "input_text": input_texts,   # list[str], e.g. ['CCC<mol>', ...]
        "smiles": smiles_list,        # list[str], original SMILES
        "molecule": molecules,        # list[dict], each with 'coordinates' and 'atom_types'
        "orig_emb1": torch.stack(orig_emb1_list)  # Tensor of shape [B, emb_dim]
    }

# ========== 7. 保存/加载方法 ==========

def save_model_parts(model, save_dir, epoch=None):
    if epoch is None:
        suffix = "latest"
    else:
        suffix = f"epoch{epoch}"
    torch.save(model.diffusion_mlp.state_dict(), os.path.join(save_dir, f"diffusion_mlp_{suffix}.pt"))
    # torch.save(model.gnn_mlp.state_dict(), os.path.join(save_dir, f"gnn_mlp_{suffix}.pt"))
    # torch.save(model.state_dict(), os.path.join(save_dir, f"model_{suffix}.pt"))
    wandb.save(os.path.join(save_dir, f"diffusion_mlp_{suffix}.pt"))
    # wandb.save(os.path.join(save_dir, f"gnn_mlp_{suffix}.pt"))
    # wandb.save(os.path.join(save_dir, f"model_{suffix}.pt"))

def load_model_parts(model, save_dir, epoch=None):
    if epoch is None:
        suffix = "latest"
    else:
        suffix = f"epoch{epoch}"
    model.diffusion_mlp.load_state_dict(torch.load(os.path.join(save_dir, f"diffusion_mlp_{suffix}.pt")))
    model.gnn_mlp.load_state_dict(torch.load(os.path.join(save_dir, f"gnn_mlp_{suffix}.pt")))
    model.load_state_dict(torch.load(os.path.join(save_dir, f"model_{suffix}.pt")))

# ========== 8. 训练循环 ==========

def train(model, dataloader, epochs=3, lr=1e-4, device='cuda', save_dir='./ckpt'):
    optimizer = torch.optim.AdamW(model.diffusion_mlp.parameters(), lr=lr)
    model.train()
    global_step = 0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch = {k: v for k, v in batch.items()}
            diffusion_emb_loss, diffusion_total_loss, coord_loss, type_loss, gnn_align_loss, total_loss = model(batch)
            diffusion_emb_loss.backward()
            optimizer.step()
            global_step += 1

            wandb.log({
                "total_loss": total_loss.item(),
                "diffusion_emb_loss": diffusion_emb_loss.item(),
                "diffusion_loss": diffusion_total_loss.item(),
                "coord_loss": coord_loss.item(),
                "type_loss": type_loss.item(),
                # "gnn_align_loss": gnn_align_loss.item(),
                "epoch": epoch,
                "step": global_step
            })

            print(f"Epoch {epoch} Step {global_step} | Total: {diffusion_emb_loss.item():.4f} | Coord: {coord_loss.item():.4f} | Type: {type_loss.item():.4f}")
        # 每epoch后分别保存
        save_model_parts(model, save_dir, epoch=epoch+1)
    # 最终保存
    save_model_parts(model, save_dir, epoch=None)
# ========== Main ==========

if __name__ == "__main__":
    llm_path = "/data1/chenyuxuan/Project/MSMLM/llama3-chem-checkpoints"
    diffusion_ckpt = "/data1/chenyuxuan/Project/MSMLM/model/diffusion/pubchem_fullatom_cond_0806_v1/log/pubchem_fullatom_cond_0806_v1/checkpoints/last-v1.ckpt"
    data_path = "/data1/chenyuxuan/Project/MSMLM/data/traindata/pubchem/train.pkl"
    batch_size = 32
    lr = 2e-5
    epochs = 5
    emb_0_dim = 3073
    emb_1_dim = 3072
    emb_2_dim = 100
    train_mode = 'diffusion'
    output_dir = "./ckpt"
    
    wandb.init(
        project="llama3-diffusion-gnn",
        name="experiment-1",
        config={
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "llm_path": llm_path
        }
    )

    # with open(data_path, "r") as f:
    #     data = json.load(f)
    import pickle as pkl
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    dataset = MolDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # gnn = GNNModel(emb_dim=100)
    gnn = None
    model = MolTrainer(llm_path, diffusion_ckpt, gnn, train_mode=train_mode, device='cuda:0')
    train(model, dataloader, epochs=epochs, lr=lr, save_dir=output_dir)
    wandb.finish()
