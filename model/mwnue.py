
import torch
import torch.nn as nn


class MWNUE(nn.Module):
  

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Predictive mean
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Predictive log variance
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """
        x:
            [batch, sequence_length, input_dim]

        Returns:
            mean:
                [batch]
            std:
                [batch]
        """

        features, _ = self.encoder(x)

        # Last time step
        h = features[:, -1, :]
        h = self.dropout(h)

        mean = self.mean_head(h).squeeze(-1)

        # Keep variance positive and numerically stable
        logvar = self.logvar_head(h).squeeze(-1)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        std = torch.exp(0.5 * logvar)

        return mean, std
