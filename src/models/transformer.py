import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATConv


class RelationEncoder(nn.Module):
    """Relation encoder using GAT to process inter-variable relationships"""

    def __init__(self, input_dim, hidden_dim, edge_index, edge_attr):
        super().__init__()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.gat = GATConv(input_dim, hidden_dim, heads=2, edge_dim=1)
        self.lin = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [B, T, D]

        Returns:
        --------
        torch.Tensor
            Output tensor with shape [B, T, H]
        """
        B, T, D = x.shape
        x = x.reshape(B * T, D)

        # GAT processing
        x = self.gat(x, self.edge_index, edge_attr=self.edge_attr)
        x = F.leaky_relu(x)
        x = self.lin(x)  # [B*T, H]

        return x.view(B, T, -1)  # [B, T, H]


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [T, B, H]

        Returns:
        --------
        torch.Tensor
            Output tensor with shape [T, B, H]
        """
        T, B, H = x.shape

        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, H, 2, device=x.device) *
                             (-np.log(10000.0) / H))

        pe = torch.zeros(T, H, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(1)  # [T, B, H] + [T, 1, H]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Time series Transformer model"""

    def __init__(self, input_dim, hidden_dim, edge_index, edge_attr, window_size):
        super().__init__()
        self.relation_encoder = RelationEncoder(input_dim, hidden_dim,
                                                edge_index, edge_attr)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=4 * hidden_dim,
                dropout=0.1,
                batch_first=False
            ),
            num_layers=3
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [B, T, D]

        Returns:
        --------
        torch.Tensor
            Output tensor with shape [B, D]
        """
        # Relation encoding
        x = self.relation_encoder(x)  # [B, T, H]
        x = x.permute(1, 0, 2)  # [T, B, H]

        # Positional encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Take last time step
        last_hidden = x[-1]  # [B, H]

        return self.predictor(last_hidden)  # [B, D]