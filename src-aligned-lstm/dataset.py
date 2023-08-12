import os
import linecache as lc

import torch
import numpy as np
from torch.utils.data import Dataset
from kipoiseq.transforms.functional import one_hot, fixed_len

# primer 1: TGCATTTTTTTCACATC
# primer 2: GGTTACGGCTGTT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform(seq):
    seq = seq[17:-13]  # remove primers
    padded_seq = fixed_len(seq, 112, "start", "X")
    one_hot_seq = one_hot(padded_seq, ["A", "C", "G", "T", "N"], "X", 0)
    return one_hot_seq


def complement_strand(strand):
    nucleotides = {65: "T", 67: "G", 71: "C", 84: "A", 78: "N", 88: "X"}
    lookup_table = ["" if i not in [65, 67, 71, 78, 84, 88] else nucleotides[i] for i in range(90)]

    comp_strand = []
    for nuc in strand:
        index = ord(nuc)
        comp_nuc = lookup_table[index]
        comp_strand.append(comp_nuc)

    comp_strand = "".join(comp_strand)
    return comp_strand

def collate_batch(batch):

    seq_list, label_list = [], []
    comp_seq_list, feature_list = [], []

    for (_sequence, _label) in batch:
        seq_list.append(transform(_sequence))
        #feature_list.append(_feature)
        label_list.append(np.array(_label, np.float32))
        #comp_seq_list.append(transform(complement_strand(_sequence)))

    sequences = np.array(seq_list + comp_seq_list)
    labels = np.array(label_list)
    #features = np.array(feature_list)

    sequences = torch.tensor(sequences).permute(0, 2, 1).float()
    labels = torch.tensor(labels).float()
    #features = torch.tensor(features).float()
    return sequences.to(device), labels.to(device)#, features.to(device)

class PromoterDataset(Dataset):

    def __init__(self, dir, filename, seq_feature_filename):
        self.dir = dir
        self.filename = filename

        with open(os.path.join(dir, filename), 'r') as fp:
            self.length = len(fp.readlines())

        self.sequence_features = np.load(seq_feature_filename)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        line = lc.getline(os.path.join(self.dir, self.filename), idx+1)
        sample = line.strip("\n").split("\t")
        seq, label = sample[0], sample[1] #self.sequence_features[idx]
        return seq, label #seq_features

