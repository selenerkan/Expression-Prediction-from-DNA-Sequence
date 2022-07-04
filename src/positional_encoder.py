import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, d_model=5, max_seq_len=112, n=10_000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float)

        # iterate through every value in our 1 hot encoded sequence
        for pos in range(max_seq_len):  # position of the current nucleotide (length of the sequence)
            for i in np.arange(int(d_model / 2)):  # position of the 1hot encoding dimension (sample number)
                denominator = np.power(n, 2 * i / d_model)
                pe[pos, 2 * i] = math.sin(pos / denominator)
                pe[pos, 2 * i + 1] = math.cos(pos / denominator)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(d_model)
        # add constant to embedding
        # seq_len = x.size(2)

        # transpose the positional encoder
        pe = torch.transpose(self.pe, 1, 2)

        x = x + pe
        return x
