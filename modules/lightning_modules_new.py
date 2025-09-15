import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_scatter import scatter_add
import numpy as np
from .equivariant_diffusion.dynamics import LigandOnlyEGNNDynamics
from .equivariant_diffusion.conditional_model_new import LigandOnlyConditionalDDPM
from .constants import FLOAT_TYPE, INT_TYPE, dataset_params
from .utils import *
from .analysis.molecule_builder import build_molecule, process_molecule


class LigandOnlyDDPM(pl.LightningModule):
    def __init__(
        self,
        diffusion_params,
        egnn_params,
        datadir,
        outdir,
        batch_size=16,
        lr=1e-4,
        num_workers=4,
        clip_grad=False,
        auxiliary_loss=False,
        loss_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        atom_encoder = dataset_params['pubchem']["atom_encoder"]
        histogram = {
            "0": 0,
            "1": 200,
            "2": 180,
            "3": 160,
            "4": 370,
            "5": 460,
            "6": 430,
            "7": 510,
            "8": 730,
            "9": 1230,
            "10": 1420,
            "11": 1450,
            "12": 1750,
            "13": 1860,
            "14": 1660,
            "15": 1380,
            "16": 1770,
            "17": 1440,
            "18": 1410,
            "19": 1630,
            "20": 1970,
            "21": 2040,
            "22": 2010,
            "23": 2120,
            "24": 1910,
            "25": 1810,
            "26": 1450,
            "27": 1270,
            "28": 1440,
            "29": 1140,
            "30": 1080,
            "31": 970,
            "32": 660,
            "33": 750,
            "34": 660,
            "35": 730,
            "36": 670,
            "37": 460,
            "38": 430,
            "39": 350,
            "40": 370,
            "41": 300,
            "42": 310,
            "43": 300,
            "44": 390,
            "45": 290,
            "46": 180,
            "47": 190,
            "48": 280,
            "49": 200,
            "50": 140,
            "51": 100,
            "52": 150,
            "53": 310,
            "54": 170,
            "55": 180,
            "56": 130,
            "57": 180,
            "58": 170,
            "59": 60,
            "60": 90,
            "61": 40,
            "62": 140,
            "63": 110,
            "64": 120,
            "65": 60,
            "66": 50,
            "67": 60,
            "68": 10,
            "69": 30,
            "70": 40,
            "71": 30,
            "72": 10,
            "73": 20,
            "74": 10,
            "75": 10,
            "76": 30,
            "77": 30,
            "78": 10,
            "79": 10,
            "80": 20,
            "81": 0,
            "82": 10,
            "83": 0,
            "84": 40,
            "85": 20,
            "86": 0,
            "87": 10,
            "88": 20,
            "89": 0,
            "90": 60,
            "91": 10,
            "92": 20,
            "93": 20,
            "94": 0,
            "95": 0,
            "96": 0,
            "97": 0,
            "98": 10,
            "99": 20,
            "100": 20,
            "101": 0,
            "102": 30,
            "103": 0,
            "104": 20,
            "105": 0,
            "106": 20,
            "107": 10,
            "108": 0,
            "109": 0,
            "110": 0,
            "111": 0,
            "112": 10
            }

        self.atom_encoder = atom_encoder
        self.atom_decoder = list(atom_encoder.keys())
        self.atom_nf = len(self.atom_decoder)
        self.x_dims = 3

        self.datadir = datadir
        self.outdir = outdir
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.clip_grad = clip_grad
        self.auxiliary_loss = auxiliary_loss
        self.T = diffusion_params.diffusion_steps
        
        hist_array = np.array(list(histogram.values()))
        self.max_num_nodes = hist_array.shape[0]

        self.net = LigandOnlyEGNNDynamics(
            atom_nf=self.atom_nf,
            n_dims=self.x_dims,
            embedding_dim=3072, 
            joint_nf=egnn_params.joint_nf,
            hidden_nf=egnn_params.hidden_nf,
            device=egnn_params.device,
            act_fn=torch.nn.SiLU(),
            n_layers=egnn_params.n_layers,
            attention=egnn_params.attention,
            condition_time=True,
            tanh=egnn_params.tanh,
            norm_constant=egnn_params.norm_constant,
            inv_sublayers=egnn_params.inv_sublayers,
            sin_embedding=egnn_params.sin_embedding,
            normalization_factor=egnn_params.normalization_factor,
            aggregation_method=egnn_params.aggregation_method,
            reflection_equivariant=egnn_params.reflection_equivariant
        )

        self.ddpm = LigandOnlyConditionalDDPM(
            dynamics=self.net,
            atom_nf=self.atom_nf,
            n_dims=self.x_dims,
            size_histogram=hist_array,
            timesteps=diffusion_params.diffusion_steps,
            loss_type=diffusion_params.diffusion_loss_type,
            norm_values=diffusion_params.normalize_factors,
            norm_biases=(None, 0.)
        )


    def configure_optimizers(self):
        return torch.optim.AdamW(self.ddpm.parameters(), lr=self.lr)

    def setup(self, stage=None):
        from dataset_new import LigandOnlyDataset
        self.train_dataset = LigandOnlyDataset(f"{self.datadir}/train.pkl", self.atom_encoder)
        self.val_dataset = LigandOnlyDataset(f"{self.datadir}/val.pkl", self.atom_encoder)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          collate_fn=self.train_dataset.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          collate_fn=self.val_dataset.collate_fn, num_workers=self.num_workers)

    def forward(self, data):
        ligand = {
            'x': data['lig_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['lig_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_lig_atoms'].to(self.device, INT_TYPE),
            'mask': data['lig_mask'].to(self.device, INT_TYPE),
        }
        embedding = data['embedding'].to(self.device, FLOAT_TYPE)  # [B, 3072]
        
        result = self.ddpm(ligand, condition=embedding, return_info=True)

        delta_log_px, error_t_lig, loss_0_x_ligand, loss_0_h, t_int, xh_lig_hat, nll, info = result

        # L2 loss during training
        denom_lig = self.x_dims * ligand['size'] + self.ddpm.atom_nf * ligand['size']
        error_t_lig = error_t_lig / denom_lig

        loss_t = 0.5 * error_t_lig
        loss_0_x_ligand = loss_0_x_ligand / (self.x_dims * ligand['size'])
        loss_0 = loss_0_x_ligand + loss_0_h

        nll = loss_t + loss_0

        if not self.training:
            nll = nll - delta_log_px

        if self.auxiliary_loss and self.training:
            x_lig_hat = xh_lig_hat[:, :self.x_dims]
            h_lig_hat = xh_lig_hat[:, self.x_dims:]
            weighted_lj_potential = self.auxiliary_weight_schedule(t_int.long()) * \
                                    self.lj_potential(x_lig_hat, h_lig_hat, ligand['mask'])
            nll = nll + weighted_lj_potential
            info['weighted_lj'] = weighted_lj_potential.mean(0)

        info.update({
            'error_t_lig': error_t_lig.mean(0),
            'loss_0': loss_0.mean(0),
            'delta_log_px': delta_log_px.mean(0),
        })
        return nll, info

    def training_step(self, data, *args):
        nll, info = self.forward(data)
        loss = nll.mean(0)
        # if loss > 2:
        #     print("Loss is too large, skipping batch")
        info['loss'] = loss
        self.log_metrics(info, 'train', batch_size=len(data['num_lig_atoms']))
        return info

    def validation_step(self, data, *args):
        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss

        self.log_metrics(info, "val", batch_size=len(data['num_lig_atoms']),
                         sync_dist=True)

        return info

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{split}/{m}', value, batch_size=batch_size, **kwargs)

    def lj_potential(self, atom_x, atom_one_hot, batch_mask):
        adj = batch_mask[:, None] == batch_mask[None, :]
        adj = adj ^ torch.diag(torch.diag(adj))
        edges = torch.where(adj)

        r = torch.sum((atom_x[edges[0]] - atom_x[edges[1]])**2, dim=1).sqrt()

        lennard_jones_radii = torch.tensor(self.lj_rm, device=r.device) / 100.0
        lennard_jones_radii = lennard_jones_radii / self.ddpm.norm_values[0]

        atom_type_idx = atom_one_hot.argmax(1)
        rm = lennard_jones_radii[atom_type_idx[edges[0]], atom_type_idx[edges[1]]]
        sigma = 2 ** (-1 / 6) * rm
        out = 4 * ((sigma / r) ** 12 - (sigma / r) ** 6)

        if self.clamp_lj is not None:
            out = torch.clamp(out, max=self.clamp_lj)

        out = scatter_add(out, edges[0], dim=0, dim_size=len(atom_x))
        return scatter_add(out, batch_mask, dim=0)
    
    @torch.no_grad()
    def generate_mol_from_embedding(
        self,
        embeddings,
        num_nodes_lig=50,
        batch_size=64,
        sanitize=False,
        largest_frag=False,
        relax_iter=0,
    ):
        """
        Generate RDKit molecules from embedding conditions.
        
        Args:
            embeddings: torch.Tensor of shape [batch_size, 3072]
            num_nodes_lig: number of ligand atoms per molecule
            batch_size: number of samples to generate
            sanitize: whether to sanitize molecules
            largest_frag: whether to return only the largest fragment
            relax_iter: steps for force field optimization
        Returns:
            A list of RDKit molecule objects.
        """
        self.eval()
        self.ddpm.eval()
        if num_nodes_lig is None:
            num_nodes_lig = self.ddpm.size_distribution.sample(batch_size)
            
        xh_lig, lig_mask = self.ddpm.sample(
            n_samples=batch_size,
            num_nodes_lig=num_nodes_lig,
            condition=embeddings
        )

        # 拆解 output todo
        x = xh_lig[:, :self.x_dims].detach().cpu() / 100
            
        atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
        lig_mask = lig_mask.cpu()

        # 构建分子
        molecules = []
        for mol_coords, mol_types in zip(
            batch_to_list(x, lig_mask),
            batch_to_list(atom_type, lig_mask)
        ):
            mol = build_molecule(mol_coords, mol_types, self.atom_decoder, add_coords=True)
            mol = process_molecule(
                mol,
                add_hydrogens=False,
                sanitize=sanitize,
                relax_iter=relax_iter,
                largest_frag=largest_frag
            )
            if mol is not None:
                molecules.append(mol)
        return molecules


class WeightSchedule:
    def __init__(self, T, max_weight, mode='linear'):
        if mode == 'linear':
            self.weights = torch.linspace(max_weight, 0, T + 1)
        elif mode == 'constant':
            self.weights = max_weight * torch.ones(T + 1)
        else:
            raise NotImplementedError(f'{mode} weight schedule is not available.')

    def __call__(self, t_array):
        return self.weights[t_array].to(t_array.device)
