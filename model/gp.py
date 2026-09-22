import torch
import torch.nn as nn
import gpytorch

# --- 1. 内部 GP 核心 ---
class InternalGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(InternalGP, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 使用 RBF 核
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 2. 外部封装类 ---
class GaussianProcess(nn.Module):
    def __init__(self, input_dim, inducing_num=200): # 1. 增加诱导点数量
        super(GaussianProcess, self).__init__()
        
        self.feat_dim = input_dim
        
        # 初始化诱导点 [-1, 1]
        inducing_points = torch.rand(inducing_num, self.feat_dim) * 2 - 1
        
        self.gp_model = InternalGP(inducing_points)
        
        # =========================================================
        # 核心修改：强制约束噪声和方差 (解决 MPIW 过大)
        # =========================================================
        
        # 2. 限制观测噪声 (Likelihood Noise)
        # 之前只有下限，模型会把噪声学得很大。
        # 现在加一个上限 0.05，强迫它认为数据很干净，必须去拟合曲线！
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(1e-6, 0.05)
        )
        # 显式初始化为很小的值
        self.likelihood.noise_covar.noise = 0.005
        
        # 3. 限制核函数幅度 (Signal Variance)
        # 初始化 outputscale 为较小值，防止初始不确定性爆炸
        self.gp_model.covar_module.outputscale = 0.01
        
    def forward(self, x):
        # 维度适配 [Batch, Seq, Feat] -> [Batch, Feat]
        if x.dim() == 3:
            x = x[:, -1, :] 
        
        # 直接返回分布对象
        dist = self.gp_model(x)
        return dist

    def get_feature_importance(self, x):
        return torch.zeros(self.feat_dim)