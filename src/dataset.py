import os
import random
from functools import reduce

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from kipoiseq.transforms.functional import one_hot, fixed_len


# primer 1: TGCATTTTTTTCACATC
# primer 2: GGTTACGGCTGTT

random.seed(13)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    sequence_list, label_list = [], []
    indices = [i for i in range(len(batch))]
    random.shuffle(indices)

    for (_sequence, _label) in batch:
        sequence_list.append(_sequence)
        label_list.append(_label)

    sequence_list = np.array([sequence_list[i] for i in indices])
    label_list = np.array([label_list[i] for i in indices])

    sequences = torch.tensor(sequence_list, dtype=torch.float64)
    labels = torch.tensor(label_list, dtype=torch.float32)
    return sequences.to(device), labels.to(device)


one_hot_encode = lambda seq: one_hot(seq, ["A", "C", "G", "T", "N"], "X", 0)
remove_primers = lambda seq: seq[17:-13]
pad_sequences = lambda seq: fixed_len(seq, 112, "start", "X")
transforms = [remove_primers, pad_sequences, one_hot_encode]


class PromoterSeqDataset(IterableDataset):

    def __init__(self, root_dir, filename, transforms):
        self.root_dir = root_dir
        self.filename = filename
        self.transforms = transforms

    def read_file(self):
        with open(os.path.join(self.root_dir, self.filename), "r") as file_obj:
            for line in file_obj:
                sample = line.strip("\n").split("\t")
                seq, label = sample[0], sample[1]
                seq, label = reduce(lambda arg, f: f(arg), self.transforms, seq), float(label)
                yield seq, label

    def __iter__(self):
        return self.read_file()
