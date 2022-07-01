import torch 
import torch.nn as nn
import math
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=5, max_seq_len = 112):
        super().__init__()
        # d_model =  sequence.shape[1]
        # max_seq_len = sequence.shape[0]
        
        # create the positional encoding matrix with    zeros
        # this matrix has shape (sequence length x encoding dimension)
        # pe = np.zeros((max_seq_len, d_model))
        pe = torch.zeros(max_seq_len, d_model)

        # iterate through every value in our 1 hot encoded sequence
        for pos in range(max_seq_len): # position of the current nucleotide
            for i in range(0, d_model - 1, 2): # position of the 1hot encoding dimension (1 to 5)
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(d_model)
        #add constant to embedding
        seq_len = x.size(2)

        # transpose the positional encoder
        self.pe = torch.transpose(self.pe, 1, 2)
        
        x = x + self.pe[:,:seq_len]
        return x