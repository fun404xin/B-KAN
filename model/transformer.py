import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # 加上位置编码
        return x + self.pe[:, :x.size(1), :]

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度 (RUL=1)
            d_model: Transformer 内部的特征维度 (Embedding size)
            nhead: 多头注意力的头数
            num_layers: Encoder 层的数量
        """
        super(Transformer, self).__init__()
        
        # 1. 特征投影层 (把输入特征映射到 d_model 维度)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        # batch_first=True 确保输入是 [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 输出层
        self.output_layer = nn.Linear(d_model, output_dim)
        
        self.d_model = d_model

    def forward(self, x):
        # 1. 维度检查与调整
        # Transformer 需要 3D 输入 [Batch, Seq_Len, Feature]
        # 如果 Dataset 给的是 2D [Batch, Feature]，我们将其视为序列长度为 1
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 2. 特征投影
        # [Batch, Seq, In_Dim] -> [Batch, Seq, d_model]
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 3. 加位置编码
        x = self.pos_encoder(x)
        
        # 4. 通过 Transformer Encoder
        # out: [Batch, Seq, d_model]
        out = self.transformer_encoder(x)
        
        # 5. 取最后一个时间步的输出进行预测
        # out[:, -1, :] shape: [Batch, d_model]
        last_step_out = out[:, -1, :]
        
        # 6. 最终预测
        prediction = self.output_layer(last_step_out)
        
        return prediction