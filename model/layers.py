import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BayesianKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, prior_sigma=1.0):
        super(BayesianKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.prior_sigma = prior_sigma

        # Grid buffer
        h = (1.0 / grid_size)
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        # Variational Parameters (Mu and Rho)
        # Base weights (Residual like)
        self.base_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline weights
        self.spline_weight_mu = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        self.spline_weight_rho = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        
        # --- 修复点：修改 scale_spline 的形状为 [out_features] ---
        # 原代码是 (out_features, in_features)，会导致广播错误
        self.scale_spline = nn.Parameter(torch.Tensor(out_features)) 

        self.reset_parameters(scale_base, scale_spline, scale_noise)
        self.kl_divergence = 0.0

    def reset_parameters(self, scale_base, scale_spline, scale_noise):
        nn.init.kaiming_uniform_(self.base_weight_mu, a=math.sqrt(5) * scale_base)
        nn.init.kaiming_uniform_(self.spline_weight_mu, a=math.sqrt(5) * scale_spline)
        nn.init.constant_(self.base_weight_rho, -5.0) # Init small variance
        nn.init.constant_(self.spline_weight_rho, -5.0)
        nn.init.constant_(self.scale_spline, 1.0)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def _bayesian_linear(self, input, mu, rho):
        sigma = torch.log1p(torch.exp(rho))
        if self.training:
            epsilon = torch.randn_like(sigma)
            weight = mu + sigma * epsilon
        else:
            weight = mu
        
        # Calculate KL for this layer
        kl = 0.5 * torch.sum((sigma**2 + mu**2) / (self.prior_sigma**2) - 1 - 2 * torch.log(sigma / self.prior_sigma))
        self.kl_divergence += kl
        return F.linear(input, weight)

    def forward(self, x):
        self.kl_divergence = 0.0
        batch_size = x.shape[0]
        
        # Base transformation
        base_output = self._bayesian_linear(F.silu(x), self.base_weight_mu, self.base_weight_rho)
        
        # Spline transformation
        spline_basis = self.b_splines(x).view(batch_size, -1)
        spline_output = self._bayesian_linear(spline_basis, self.spline_weight_mu, self.spline_weight_rho)
        
        # --- 修复点：确保广播正确 ---
        # spline_output [B, out] * scale_spline [out] -> 自动广播为 [B, out]
        output = base_output + spline_output * self.scale_spline
        
        return output