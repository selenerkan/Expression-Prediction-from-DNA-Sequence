import os
import sys

def get_metadata():
    counter = 0
    index, max_len = 0, 0

    all_sequences = []
    file_obj = open("/Users/goktug/PycharmProjects/ML4RG Project/data/train_sequences.txt")

    for line in file_obj:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            all_sequences.append(seq)

        if max_len < len(seq):
            max_len = len(seq)
            index = counter

        counter += 1

    return all_sequences, max_len, index


sequences, max_len, index = get_metadata()
total_bytes = sys.getsizeof(sequences)

print("Maximum sequence length: ", max_len)
print("Index of longest sequence: ", index)
print("Total Allocated Byte: ", total_bytes)
print("Total Allocated GigaByte: ", total_bytes / 10 ** 6)