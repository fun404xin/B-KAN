import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BayesianKANLayer

class BayesianKAN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8], grid_size=5, spline_order=3):
        super(BayesianKAN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.input_dim = input_dim # 记录输入维度
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            self.layers.append(BayesianKANLayer(in_dim, h_dim, grid_size=grid_size, spline_order=spline_order))
            in_dim = h_dim
            
        self.mu_head = nn.Linear(in_dim, 1)
        self.logvar_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        total_kl = 0.0
        for layer in self.layers:
            x = layer(x)
            total_kl += layer.kl_divergence
            
        mu = self.mu_head(x)
        # 限制 logvar 防止方差爆炸
        logvar = self.logvar_head(x)
        logvar = torch.clamp(logvar, min=-4, max=1) 
        
        return mu, logvar, total_kl

    def get_feature_importance(self, x):
        """
        计算输入特征的动态贡献度 (Dynamic Feature Attribution)
        对应论文公式: S_k(t) = || phi_k(x_k^t) ||_2
        """
        # 我们只关心第一层，因为那是直接与输入特征相连的层
        layer0 = self.layers[0]
        batch_size = x.size(0)
        
        # 存储每个特征的贡献度 [Batch, Input_Dim]
        contributions = torch.zeros(batch_size, self.input_dim).to(x.device)
        
        # 1. 准备 B-Spline 基函数 [Batch, In, Coeffs]
        # 注意：这里需要调用 layer0 内部的 b_splines 方法
        spline_basis = layer0.b_splines(x) 
        
        # 2. 遍历每一个输入特征 k
        for k in range(self.input_dim):
            # --- A. 计算 Spline 部分的贡献 ---
            # 取出第 k 个特征对应的基函数: [Batch, Grid+Order]
            basis_k = spline_basis[:, k, :] 
            
            # 取出第 k 个特征对应的 Spline 权重: [Out, Grid+Order]
            # 权重形状是 [Out, In * Coeffs]，需要切片
            coeff_len = layer0.grid_size + layer0.spline_order
            start_idx = k * coeff_len
            end_idx = (k + 1) * coeff_len
            
            # 使用均值权重进行解释 (忽略方差，关注期望贡献)
            weight_spline_k = layer0.spline_weight_mu[:, start_idx:end_idx] 
            
            # 线性变换: Basis [B, C] @ Weight.T [C, Out] -> [B, Out]
            spline_out_k = F.linear(basis_k, weight_spline_k)
            
            # 乘以缩放系数 (广播)
            # scale_spline 是 [Out]
            spline_out_k = spline_out_k * layer0.scale_spline
            
            # --- B. 计算 Base (SiLU) 部分的贡献 ---
            # x[:, k] 形状 [Batch] -> [Batch, 1]
            x_k = x[:, k].unsqueeze(1)
            # Base 权重: [Out, In] -> 取第 k 列 -> [Out, 1]
            weight_base_k = layer0.base_weight_mu[:, k].unsqueeze(1)
            
            # F.linear(x, A) = x @ A.T
            # 这里手动算: SiLU(x_k) * w_k
            base_out_k = F.silu(x_k) @ weight_base_k.t() # [Batch, Out]
            
            # --- C. 总激活输出 ---
            total_out_k = base_out_k + spline_out_k # [Batch, Out]
            
            # --- D. 计算 L2 范数作为贡献度 ---
            # dim=1 表示在该特征触发的所有隐藏层神经元上的总响应强度
            score_k = torch.norm(total_out_k, p=2, dim=1) # [Batch]
            
            contributions[:, k] = score_k
            
        return contributions