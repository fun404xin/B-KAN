import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.0):
        super(LSTM, self).__init__()
        # batch_first=True 确保输入格式为 (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 1. 检查输入维度，如果是 2D [Batch, Feature]，则强转为 3D [Batch, 1, Feature]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 2. LSTM 前向传播
        # out shape: [Batch, Seq_Len, Hidden_Dim]
        out, _ = self.lstm(x)
        
        # 3. 取最后一个时间步的输出
        # out[:, -1, :] 意思是：取所有 Batch，取最后一个时间步，取所有 Hidden特征
        last_step_out = out[:, -1, :]
        
        # 4. 全连接层预测
        prediction = self.fc(last_step_out)
        return prediction