import torch
import torch.nn as nn
import math

class MCDropoutTransformer(nn.Module):
    """
    MC-Dropout Transformer for Probabilistic RUL Prediction
    
    机制:
    1. 训练时: Dropout 开启，输出 mu 和 logvar，使用 Gaussian NLL Loss (ELBO) 训练。
    2. 推理时: 保持 Dropout 开启 (通过 model.train())，进行多次前向传播。
       - mu 的波动代表模型不确定性 (Epistemic)。
       - logvar 代表数据不确定性 (Aleatoric)。
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_dim=1):
        super(MCDropoutTransformer, self).__init__()
        
        # 1. Embedding & Positional Encoding
        self.embedding = nn.Linear(input_dim, d_model)
        # 简单的可学习位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model)) 
        
        # 2. Transformer Encoder (自带 Dropout)
        # batch_first=True 适配 [Batch, Seq, Feat]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout,  # 这里的 Dropout 是 MC-Dropout 的核心
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 输出头 (Heteroscedastic Regression Heads)
        # 均值头
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout), # 输出层前也加 Dropout
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 方差头 (预测 log(sigma^2))
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        """
        Returns:
            mu: [Batch, 1]
            logvar: [Batch, 1]
            kl: 0.0 (MC-Dropout 不需要显式 KL，由 Weight Decay 替代)
        """
        # --- 1. 维度适配 ---
        # 如果输入是 [Batch, Features] (2D)，扩展为 [Batch, 1, Features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.shape
        
        # --- 2. Embedding + Pos ---
        # 截取对应长度的位置编码
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # --- 3. Transformer Forward ---
        # 这里的 Dropout 在 model.train() 模式下会随机 drop
        x = self.transformer_encoder(x)
        
        # --- 4. Pooling ---
        # 取最后一个时间步 (RUL 预测通常关注最新状态)
        x = x[:, -1, :] 
        
        # --- 5. Heads ---
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        
        # 返回 0.0 作为 KL，适配 elbo_loss 接口
        return mu, logvar, torch.tensor(0.0).to(x.device)

    def get_feature_importance(self, x):
        """ 基于梯度的简单的特征重要性分析 """
        x.requires_grad_(True)
        mu, _, _ = self.forward(x)
        mu.sum().backward()
        # [Batch, Seq, Feat] -> [Feat]
        if x.dim() == 3:
            return x.grad.abs().mean(dim=0).mean(dim=0)
        return x.grad.abs().mean(dim=0)