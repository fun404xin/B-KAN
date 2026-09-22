import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLSTM(nn.Module):
    """
    贝叶斯 LSTM 基准模型
    修复了输入维度问题：自动将 2D 特征输入转换为 LSTM 需要的 3D 时序输入。
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(BayesianLSTM, self).__init__()
        
        # 定义 LSTM 层
        # batch_first=True 意味着输入形状应为 [Batch, Seq_Len, Features]
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout_rate)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出头
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # --- 修复核心 ---
        # 1. 检查输入维度
        # Data Loader 给出的 x 是 [Batch, Features] (2D)
        # LSTM 需要 [Batch, Seq_Len, Features] (3D)
        if x.dim() == 2:
            x = x.unsqueeze(1) # 在中间插入维度，变成 [Batch, 1, Features]
            
        # 2. 通过 LSTM
        # out 形状: [Batch, Seq_Len, Hidden_Dim]
        out, _ = self.lstm(x)
        
        # 3. 取最后一个时间步
        # 此时 out 是 3D 的，切片操作不会报错
        out = out[:, -1, :] # 形状变为 [Batch, Hidden_Dim]
        
        # 4. Dropout (MC Sampling)
        out = self.dropout(out)
        
        # 5. 输出均值和方差
        mu = self.mu_head(out)
        logvar = self.logvar_head(out)
        
        # LSTM 基准通常不计算显式的 KL 散度 (使用 Dropout 近似贝叶斯)，返回 0
        return mu, logvar, torch.tensor(0.0).to(x.device)