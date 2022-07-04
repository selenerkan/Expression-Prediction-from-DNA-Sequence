import torch
import torch.nn as nn
import math
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model=5, dropout=0.1, max_len=112):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        pe = torch.permute(pe, (1, 2, 0))
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # pe = torch.transpose(self.pe, 1, 2) [1024, 5, 512] [1, 512, 5]
        x = x + self.pe
        return self.dropout(x)
