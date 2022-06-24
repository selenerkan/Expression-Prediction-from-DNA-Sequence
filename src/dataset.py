import os
import random
from functools import reduce
import linecache as lc

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, Dataset
from kipoiseq.transforms.functional import one_hot, fixed_len


# primer 1: TGCATTTTTTTCACATC
# primer 2: GGTTACGGCTGTT

random.seed(13)
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
    comp_seq_list, comp_label_list = [], []

    indices = [i for i in range(len(batch))]
    random.shuffle(indices)
    print("Batch Size: ", len(batch))
    for (_sequence, _label) in batch:
        seq_list.append(transform(_sequence))
        label_list.append(np.array(_label, np.float32))
        comp_seq_list.append(transform(complement_strand(_sequence)))

    sequences = np.array(seq_list + comp_seq_list)
    labels = np.array(label_list)

    sequences = torch.tensor(sequences).permute(0, 2, 1).float()
    labels = torch.tensor(labels).float()
    return sequences.to(device), labels.to(device)


class PromoterDataset(Dataset):

    def __init__(self, dir, filename):
        self.dir = dir
        self.filename = filename

        with open(os.path.join(dir, filename), 'r') as fp:
            self.length = len(fp.readlines())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        line = lc.getline(os.path.join(self.dir, self.filename), idx+1)
        sample = line.strip("\n").split("\t")
        seq, label = sample[0], sample[1]
        return seq, label

