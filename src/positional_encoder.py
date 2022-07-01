import torch
import math
import torch.nn as nn
from torch.autograd import Variable

# embedding is not needed because we already have 1hot encoding

# POSITIONAL ENCODER
# we have an encoded matrix of gene sequence with size (5 x sequence_length)
# positional encoder will get three parameters 
    # pos: position of the nucleotide in the sequence
    # i: position in the encoded direction (form 0 to 5 in our case)
    # d_model: dimension of the encoded dim (5 for us)

# for each 

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 120):
        super().__init__()

        # set the length of the encoding dimension
        self.d_model = d_model

        # create the positional encoding matrix with zeros
        # this matrix has shape (sequence length x encoding dimension)
        pe = torch.zeros(max_seq_len, d_model)

        # iterate through every value in our 1 hot encoded sequence
        for pos in range(max_seq_len): # position of the current nucleotide
            for i in range(0, d_model, 2): # position of the 1hot encoding dimension (1 to 5)
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        # pe = pe.unsqueeze(0) # creates extra dimension, creates batch of size 1 with the goven input
        # self.register_buffer('pe', pe) 
        #If you have parameters in your model, which should be saved and restored in the state_dict, 
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), 
        # so that the optimizer won’t have a change to update them.
 
    
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)

        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x