"""
Discrete noise schedule. The code below is a modified version of DiGress: https://arxiv.org/abs/2209.14734
"""
import numpy as np
import torch
from torch.nn import functional as F

from building_diffusion.constants import AVERAGE_NUM_NODES
from building_diffusion.utils.seed import allow_nondeterministic


class NoiseScheduler(torch.nn.Module):
    """
    Discrete noise of the voxel labels (DiGress): with probability 1 - alpha_bar_t, a label is resampled
    uniformly over the x_classes classes. Timesteps can be given as integers t or normalized t / timesteps.
    """
    def __init__(self, timesteps, x_classes, device):
        super(NoiseScheduler, self).__init__()
        self.timesteps = timesteps
        self.X_classes = x_classes
        
        self.device = device
        self.betas = self.cosine_beta_schedule_discrete(timesteps).to(self.device)
        self.register_buffer('betas_buffer', self.betas.float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999).to(self.device)
        log_alpha = torch.log(self.alphas).to(self.device)
        # cumsum has no deterministic CUDA kernel, but its result on these 500 values is repeatable. It is kept
        # on the same device as before (CPU and GPU round the last bit differently), so the schedule is unchanged
        with allow_nondeterministic():
            log_alpha_bar = torch.cumsum(log_alpha, dim=0).to(self.device)
        self.alphas_bar = torch.exp(log_alpha_bar).to(self.device)

        self.u_x = (torch.ones(1, self.X_classes, self.X_classes) / self.X_classes).to(self.device)

    def cosine_beta_schedule_discrete(self, timesteps, average_num_nodes=AVERAGE_NUM_NODES, s=0.008):
        """
        Cosine noise schedule (timesteps + 1 betas), with the lower bound on the betas used by DiGress.
        average_num_nodes: average graph size of the dataset (AVERAGE_NUM_NODES for Building-GAN).
        """
        steps = timesteps + 2
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = 1 - alphas

        assert timesteps >= 100

        # Lower bound on the betas, from DiGress (https://arxiv.org/abs/2209.14734), where it is computed
        # for the edges of a graph: with p = 1 - 1 / num_edge_classes (2 edge classes: edge / no edge) and
        # num_edges = n (n - 1) / 2 node pairs, a beta of updates_per_graph / (p * num_edges) changes about
        # updates_per_graph edges per graph and step. It is kept unchanged here, although only node labels are noised.
        p = 1 / 2       # 1 - 1 / num_edge_classes 
        num_edges = average_num_nodes * (average_num_nodes - 1) / 2

        # First 100 steps: only a few updates per graph
        updates_per_graph = 1.2
        beta_first = updates_per_graph / (p * num_edges)

        betas[betas < beta_first] = beta_first

        return torch.from_numpy(betas).float()

    def get_Qt(self, beta_t):
        """One-step transition matrices Q_t from betas (bs x 1): bs x K x K."""
        beta_t = beta_t.unsqueeze(1).to(self.device)
        self.u_x = self.u_x.to(self.device)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=self.device).unsqueeze(0)
        return q_x.to(self.device)
    
    def get_Qt_bar(self, alpha_bar_t):
        """Cumulative transition matrices Q_bar_t from alpha_bar_t (bs x 1): bs x K x K."""
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(self.device)
        self.u_x = self.u_x.to(self.device)
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=self.device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        return q_x.to(self.device)

    def forward(self, t_normalized=None, t_int=None):
        """Beta of a timestep, given as t_normalized or t_int (one of them); same shape as the timestep."""
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas_buffer[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        """alpha_bar of a timestep, given as t_normalized or t_int (one of them); same shape as the timestep."""
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]

    def sample_discrete_features(self, probX):
        """Sample one class per node from class probabilities (bs x n x K): class indices (bs x n)."""
        bs, n, _ = probX.shape
        probX = probX.reshape(bs * n, -1)
        X_t = probX.multinomial(1).reshape(bs, n)
        return X_t.to(self.device)

    def apply_noise(self, X, timestep = None):
        """
        Noisy labels at a random timestep in [1, timesteps) (or the given timestep), from one-hot labels X (N x K).
        Returns the timestep (1 x 1) and the noisy one-hot labels (1 x N x K).
        """
        X = X.to(self.device)

        # Random timestep in [1, timesteps), unless a timestep is given
        if timestep is None:
            t_int = torch.randint(1, self.timesteps, size=(1, 1), device=X.device).float()
        else:
            t_int = torch.tensor(timestep).float().unsqueeze(0).unsqueeze(1).to(self.device)

        t_float = (t_int / self.timesteps).clone().detach().to(self.device)
        alpha_t_bar = self.get_alpha_bar(t_normalized=t_float)

        Qtb = self.get_Qt_bar(alpha_t_bar)
        probX = X @ Qtb

        sampled_t = self.sample_discrete_features(probX=probX)
        X_t = F.one_hot(sampled_t, num_classes=self.X_classes)

        # Return Timestep, and Noisy Graph at Timestep t        
        return t_int.to(self.device), X_t.to(self.device)
