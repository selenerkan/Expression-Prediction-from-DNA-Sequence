import random
from sklearn.model_selection import train_test_split
import numpy as np


dir1 = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_sequences.txt"
dir2 = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_comp_sequences.txt"
dir3 = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_missing_sequences.txt"
dir4 = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_subsequences.txt"
dir5 = "/Users/goktug/PycharmProjects/ML4RG Project/data/valid_subsequences.txt"


def complete_sequences(dir1, dir2):

    file1 = open(dir1, "r")
    file2 = open(dir2, "w")

    for line in file1:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            if "N" not in seq:
                file2.write(seq + "\t" + label + "\n")

    file1.close()
    file2.close()


def missing_sequences(dir1, dir2):
    file1 = open(dir1, "r")
    file2 = open(dir2, "w")

    for line in file1:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            if "N" in seq:
                file2.write(seq + "\t" + label + "\n")

    file1.close()
    file2.close()


def create_sub_dataset(num_comp_seq, num_miss_seq, dir1, dir2, dir3, dir4, train_ratio):

    comp_sequences = []
    miss_sequences = []

    file3 = open(dir3, "w")
    file4 = open(dir4, "w")

    if num_comp_seq > 0:
        file1 = open(dir1, "r")
        comp_sequences = file1.readlines()

        random_indices = np.random.choice(range(len(comp_sequences)), num_comp_seq)
        comp_sequences = [comp_sequences[i] for i in random_indices]

        file1.close()

    if num_miss_seq > 0:

        file2 = open(dir2, "r")
        miss_sequences = file2.readlines()

        random_indices = np.random.choice(range(len(miss_sequences)), num_miss_seq)
        miss_sequences = [miss_sequences[i] for i in random_indices]

        file2.close()

    sequences = comp_sequences + miss_sequences
    random.shuffle(sequences)

    seq_train, seq_test, label_train, label_test = train_test_split(sequences, [i for i in range(len(sequences))],
                                                                    train_size=train_ratio)

    for seq in seq_train:
        file3.write(seq)

    for seq in seq_test:
        file4.write(seq)

    file3.close()
    file4.close()


#complete_sequences(dir1, dir2)
#missing_sequences(dir1, dir3)
create_sub_dataset(200_000, 0, dir2, dir3, dir4, dir5, 0.8)
