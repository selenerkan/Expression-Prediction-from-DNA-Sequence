import numpy as np

def sequence_counter(file_ptr):

    sequences = [0 for i in range(18)]

    for line in file_ptr:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            index = int(float(label))
            sequences[index] += 1

    return sequences

file_ptr = open("../data/train_subsequences_balanced.txt")
print(sequence_counter(file_ptr))
file_ptr.close()