# ligand_only_ddpm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
import math
import numpy as np
from typing import Dict


class LigandOnlyConditionalDDPM(nn.Module):
    def __init__(
        self,
        dynamics: nn.Module,
        atom_nf: int,
        n_dims: int,
        size_histogram:Dict,
        timesteps: int = 1000,
        loss_type='vlb',
        noise_schedule='learned',
        noise_precision=1e-4,
        norm_values=(1., 1.),
        norm_biases=(None, 0.),
    ):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.dynamics = dynamics
        self.atom_nf = atom_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))
        self.size_distribution = DistributionNodes1D(size_histogram)
        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule,
                                                 timesteps=timesteps,
                                                 precision=noise_precision)

    def log_pN(self, batch_n_nodes):
        return self.size_distribution.log_prob(batch_n_nodes)
        
    def log_pxh_given_z0_without_constants(self, ligand, z_0_lig, eps_lig, net_out_lig, gamma_0, epsilon=1e-10):
        # 只对 position 部分进行 log-likelihood 估计（Gaussian）
        eps_x = eps_lig[:, :self.n_dims]
        net_x = net_out_lig[:, :self.n_dims]

        # 只对 h（atom type one-hot）部分进行分类 log prob（离散高斯积分）
        z_h = z_0_lig[:, self.n_dims:]
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]
        estimated_onehot = z_h * self.norm_values[1] + self.norm_biases[1]
        centered = estimated_onehot - 1  # one-hot 类别中心在1

        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        # log P(x|z0) 部分（连续值）
        log_p_x = -0.5 * self.sum_except_batch((eps_x - net_x) ** 2, ligand['mask'])

        # log P(h|z0) 部分（分类）
        log_ph_cat = torch.log(
            self.cdf_standard_gaussian((centered + 0.5) / sigma_0_cat[ligand['mask']]) -
            self.cdf_standard_gaussian((centered - 0.5) / sigma_0_cat[ligand['mask']]) +
            epsilon
        )
        log_Z = torch.logsumexp(log_ph_cat, dim=1, keepdim=True)
        log_prob = log_ph_cat - log_Z
        log_ph = self.sum_except_batch(log_prob * ligand_onehot, ligand['mask'])

        return log_p_x, log_ph
    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * \
               np.log(self.norm_values[0])

    def subspace_dimensionality(self, input_size):
        """Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined."""
        return (input_size - 1) * self.n_dims
    
    def forward(self, ligand, condition=None, return_info=False):
        ligand = self.normalize(ligand)

        # 1. delta_log_px: volume change
        delta_log_px = self.delta_log_px(ligand['size'])

        # 2. Sample timestep
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(ligand['size'].size(0), 1), device=ligand['x'].device).float()
        t = t_int / self.T
        s_int = t_int - 1
        s = s_int / self.T
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # 3. gamma embeddings
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])
        gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])

        # 4. build [x, h]
        xh_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)

        # 5. noise
        z_t_lig, eps_t_lig = self.noised_representation(xh_lig, ligand['mask'], gamma_t)

        # 6. predict
        net_out_lig = self.dynamics(z_t_lig, t, ligand['mask'], condition=condition)

        # 7. reconstruct for auxiliary LJ loss
        xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t, ligand['mask'])

        # 8. L2 error
        error_t_lig = self.sum_except_batch((eps_t_lig - net_out_lig) ** 2, ligand['mask'])

        # 9. SNR weight
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)

        if self.training:
            # Compute log p(x,h|z_0) (no constant)
            log_px_z0, log_ph_z0 = self.log_pxh_given_z0_without_constants(ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t)
            loss_0_x_ligand = -log_px_z0 * t_is_zero.squeeze()
            loss_0_h = -log_ph_z0 * t_is_zero.squeeze()

            # Mask loss_t only when t ≠ 0
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()
        else:
            # For evaluation, t=0 loss
            t_zeros = torch.zeros_like(t)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])
            z_0_lig, eps_0_lig = self.noised_representation(xh_lig, ligand['mask'], gamma_0)
            net_out_0_lig = self.dynamics(z_t_lig, t, ligand['mask'], condition=condition)

            log_px_z0, log_ph_z0 = self.log_pxh_given_z0_without_constants(ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0)
            loss_0_x_ligand = -log_px_z0
            loss_0_h = -log_ph_z0

        # 10. KL prior
        kl_prior = self.kl_prior(xh_lig, ligand['mask'], ligand['size'])

        # 11. constant from Gaussian likelihood
        neg_log_constants = -self.log_constants_p_x_given_z0(ligand['size'], device=error_t_lig.device)

        # 12. log_pN
        log_pN = self.log_pN(ligand['size'])

        # 13. Compose losses
        loss_0 = loss_0_x_ligand + loss_0_h + neg_log_constants
        loss_t = -self.T * 0.5 * SNR_weight * error_t_lig
        nll = loss_0 + loss_t + kl_prior

        if not self.training:
            nll = nll - delta_log_px - log_pN


        lj_mean = torch.tensor(0.0, device=ligand['x'].device)

        # 14. Logging info
        info = {
            'eps_hat_lig_x': scatter_mean(net_out_lig[:, :self.n_dims].abs().mean(1), ligand['mask'], dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(net_out_lig[:, self.n_dims:].abs().mean(1), ligand['mask'], dim=0).mean(),
            'loss_0': loss_0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'delta_log_px': delta_log_px.mean(0),
            'log_pN': log_pN.mean(0),
            'SNR_weight': SNR_weight.mean(0),
            'weighted_lj': lj_mean
        }

        return delta_log_px, error_t_lig, loss_0_x_ligand, loss_0_h, t_int, xh_lig_hat, nll, info

    @staticmethod
    def SNR(gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice
        negligible in the loss. However, you compute it so that you see it when
        you've made a mistake in your noise schedule.
        """
        batch_size = len(num_nodes)

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        # Compute means.
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = \
            mu_T_lig[:, :self.n_dims], mu_T_lig[:, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        # Compute KL for h-part.
        zeros = torch.zeros_like(mu_T_lig_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2 = self.sum_except_batch((mu_T_lig_h - zeros) ** 2, mask_lig)
        kl_distance_h = self.gaussian_KL(mu_norm2, sigma_T_h, ones, d=1)

        # Compute KL for x-part.
        zeros = torch.zeros_like(mu_T_lig_x)
        ones = torch.ones_like(sigma_T_x)
        mu_norm2 = self.sum_except_batch((mu_T_lig_x - zeros) ** 2, mask_lig)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2, sigma_T_x, ones, subspace_d)

        return kl_distance_x + kl_distance_h
    def log_constants_p_x_given_z0(self, n_nodes, device):
        """Computes p(x|z0)."""

        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes)

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean of
                    distribution q and distribution p: ||mu_q - mu_p||^2
                q_sigma: Standard deviation of distribution q.
                p_sigma: Standard deviation of distribution p.
                d: dimension
            Returns:
                The KL distance
            """
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / \
               (p_sigma ** 2) - 0.5 * d

    def noised_representation(self, xh_lig, lig_mask, gamma_t):
        eps_lig = self.sample_gaussian(size=xh_lig.shape, device=xh_lig.device)
        z_t_lig = xh_lig + eps_lig  # simplified noise injection
        return z_t_lig[lig_mask], eps_lig[lig_mask]

    def normalize(self, ligand):
        ligand['x'] = ligand['x'] / self.norm_values[0]
        ligand['one_hot'] = (ligand['one_hot'].float() - self.norm_biases[1]) / self.norm_values[1]
        return ligand

    def inflate_batch_array(self, array, target):
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sum_except_batch(self, x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)

    def sample_gaussian(self, size, device):
        return torch.randn(size, device=device)

    def sigma(self, gamma, target_tensor):
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        return z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / alpha_t[batch_mask]

    def alpha(self, gamma, target_tensor):
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def unnormalize(self, x, h_cat):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        return x, h_cat

    def cdf_standard_gaussian(self, x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def remove_mean_batch(x, indices):
        mean = scatter_mean(x, indices, dim=0)
        return x - mean[indices]


    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor,
                                  gamma_s: torch.Tensor,
                                  target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    @torch.no_grad()
    def sample(self, n_samples, num_nodes_lig, condition=None):
        device = condition.device if condition is not None else torch.device('cpu')
        # total_nodes = n_samples * num_nodes_lig
        # lig_mask = torch.arange(total_nodes, device=device) // num_nodes_lig
        total_nodes = torch.sum(num_nodes_lig)
        lig_mask = torch.arange(n_samples, device=device).repeat_interleave(torch.tensor(num_nodes_lig, device=device))
        
        # 初始化为标准正态噪声
        z_lig = torch.randn((total_nodes, self.n_dims + self.atom_nf), device=device)

        for t_inv in reversed(range(self.T)):
            t_array = torch.full((n_samples, 1), fill_value=(t_inv + 1), device=device) / self.T
            s_array = torch.full((n_samples, 1), fill_value=t_inv, device=device) / self.T

            gamma_s = self.gamma(s_array)
            gamma_t = self.gamma(t_array)

            sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_lig)
            alpha_ts = self.inflate_batch_array(alpha_t_given_s, z_lig)[lig_mask]
            sigma_ts = self.inflate_batch_array(sigma_t_given_s, z_lig)[lig_mask]
            sigma2_ts = self.inflate_batch_array(sigma2_t_given_s, z_lig)[lig_mask]

            # dynamics model predicts noise
            eps_hat = self.dynamics(z_lig, t_array, lig_mask, condition=condition)

            z_pos = z_lig[:, :self.n_dims] / alpha_ts - (sigma2_ts / alpha_ts / self.sigma(self.inflate_batch_array(gamma_t, z_lig), z_lig)[lig_mask]) * eps_hat[:, :self.n_dims]
            z_lig = torch.cat([
                self.remove_mean_batch(z_pos, lig_mask).to(z_lig.dtype),
                z_lig[:, self.n_dims:]
            ], dim=1)

        # 最后一步，预测 x/h
        t_zeros = torch.zeros((n_samples, 1), device=device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), z_lig)
        net_out = self.dynamics(z_lig, t_zeros, lig_mask, condition=condition)
        xh_lig = self.xh_given_zt_and_epsilon(z_lig, net_out, gamma_0, lig_mask)

        # Unnormalize x, h_cat
        x, h_cat = self.unnormalize(xh_lig[:, :self.n_dims], xh_lig[:, self.n_dims:])
        xh_lig = torch.cat([x, h_cat], dim=1)

        # 将最终预测写入 out_lig
        out_lig = torch.zeros((1,) + xh_lig.size(), device=xh_lig.device)
        out_lig[0] = xh_lig

        return out_lig.squeeze(0), lig_mask

class DistributionNodes1D:
    def __init__(self, histogram, node_values=None):
        """
        单变量原子数概率分布构造器

        Args:
            histogram: list or array of frequencies/probabilities for each possible atom count
            node_values: list of actual atom count values (optional).
                         If None, assumes node_values = [0, 1, ..., len(histogram)-1]
        """
        histogram = torch.tensor(histogram, dtype=torch.float32)
        histogram = histogram + 1e-3  # for numerical stability
        prob = histogram / histogram.sum()
        self.prob = prob

        self.node_values = (
            torch.tensor(node_values, dtype=torch.long)
            if node_values is not None
            else torch.arange(len(prob), dtype=torch.long)
        )

        self.min_val = self.node_values.min().item()
        self.max_val = self.node_values.max().item()

        # Map: atom_count -> index
        self.node_value_to_index = {
            int(v.item()): i for i, v in enumerate(self.node_values)
        }

        # Categorical distribution
        self.m = torch.distributions.Categorical(probs=self.prob, validate_args=False)

    def sample(self, n_samples=1):
        """
        从分布中采样原子数（真实值）
        Returns: Tensor of shape [n_samples]
        """
        idx = self.m.sample((n_samples,))
        return self.node_values[idx]

    def log_prob(self, batch_n_nodes_1: torch.Tensor) -> torch.Tensor:
        """
        计算输入原子数的对数概率（超出支持范围的会被 clamp）

        Args:
            batch_n_nodes_1: Tensor of shape [B], 每个元素是实际原子数（如 22, 25, 30）

        Returns:
            Tensor of shape [B], 每个元素是对应的 log prob
        """
        assert batch_n_nodes_1.ndim == 1, "Input must be a 1D tensor."
        device = batch_n_nodes_1.device
        batch_n_nodes_1 = batch_n_nodes_1.to(self.m.logits.device)
        # Clamp 到支持范围
        clamped = torch.clamp(batch_n_nodes_1, min=self.min_val, max=self.max_val)

        # 记录哪些超出了范围
        if (batch_n_nodes_1 != clamped).any():
            print(f"[Warning] Some values out of supported range [{self.min_val}, {self.max_val}]. Clamped.")

        # 映射到 index
        idx = torch.tensor(
            [self.node_value_to_index[int(v.item())] for v in clamped],
            device=clamped.device
        )

        return self.m.log_prob(idx).to(device)

    def entropy(self):
        """
        返回当前分布的信息熵 H[p]（越大表示越分散）
        """
        return self.m.entropy().item()

    def to(self, device):
        """
        Optional: 用于模型中将分布移到指定设备（GPU / CPU）
        """
        self.prob = self.prob.to(device)
        self.node_values = self.node_values.to(device)
        self.m = torch.distributions.Categorical(probs=self.prob)
        return self
    
class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function.
    Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2



class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)



class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined
    (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
