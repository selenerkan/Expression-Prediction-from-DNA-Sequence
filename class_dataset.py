
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
    comp_seq_list, comp_label_list = [], []

    for (_sequence, _label) in batch:
        seq_list.append(transform(_sequence))
        label_list.append(np.array(_label, np.float32))
        comp_seq_list.append(transform(complement_strand(_sequence)))

    sequences = np.array(seq_list + comp_seq_list)
    labels = np.array(label_list)

    sequences = torch.tensor(sequences).permute(0, 2, 1).float()
    labels = torch.tensor(labels).float()
    return sequences.to(device), labels.to(device)

def GetLabels(num_classes = 500):
    """parses through the training data and returns the frequently occurring 500 classes"""

    n = 300000 #Parse ever n lines at a time
    filename = "./data/train_sequences.txt"
    classes = np.array([])
    with open(filename) as my_file:
        num_lines = len(my_file.readlines()) #Maximum lines in the file
    my_file.close()
    with open(filename) as my_file:
        line = 0
        while line < num_lines:

            head = np.array([next(my_file).strip('\n').replace('\t', ' ').split(' ') for x in range(line, line+n)])
            classes = np.append(classes, head[:,1])
            line = line+n

            if line + n > num_lines:
                n = num_lines - line
    #file = np.genfromtxt(filename, max_rows=3)
    unique, counts = np.unique(classes, return_counts=True)
    idx = counts.argsort()
    classes_new = unique[idx]
    top_classes = classes_new[-1*num_classes:].astype(np.float64)
    return top_classes

class PromoterDataset(Dataset):

    def __init__(self, dir, filename,classes):
        self.dir = dir
        self.filename = filename
        self.classes = classes

        with open(os.path.join(dir, filename), 'r') as fp:
            self.length = len(fp.readlines())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):


        line = lc.getline(os.path.join(self.dir, self.filename), idx+1)
        sample = line.strip("\n").split("\t")
        seq, label_temp = sample[0], sample[1]
        difference_array = np.absolute(self.classes-label_temp)
        label_temp = difference_array.argmin() #index of the class
        label = np.zeros(500)
        label[label_temp] = 1
        return seq, label

if __name__=="__main__":
    import time
    start = time.time()
    classes = GetLabels(500)
    end = time.time()
    print(classes)
    print("labels got in:",end-start)