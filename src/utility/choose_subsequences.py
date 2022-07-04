from sklearn.model_selection import train_test_split
import numpy as np

"""
train_dir = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_sequences.txt"
train_comp_dir = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_comp_sequences.txt"
train_miss_dir = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_missing_sequences.txt"
train_subset_dir = "/Users/goktug/PycharmProjects/ML4RG Project/data/train_subsequences.txt"
valid_subset_dir = "/Users/goktug/PycharmProjects/ML4RG Project/data/valid_subsequences.txt" """


def complete_sequences(train_dir, train_comp_dir):

    """
    * It reads all sequences in train dataset, choose the ones without missing nucleotide, and write them to the file
    "train_comp_sequences.txt".
    It takes the directories of "train_sequences.txt" and "train_comp_sequences.txt" as input arguments.
    """

    file1 = open(train_dir, "r")
    file2 = open(train_comp_dir, "w")

    for line in file1:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            if "N" not in seq:
                file2.write(seq + "\t" + label + "\n")

    file1.close()
    file2.close()


def missing_sequences(train_dir, train_miss_dir):
    """
    * It reads all sequences in train dataset, choose the ones with missing nucleotide, and write them to the file
    "train_missing_sequences.txt".
    * It takes the directories of "train_sequences.txt" and "train_missing_sequences.txt" as input arguments.
    """

    file1 = open(train_dir, "r")
    file2 = open(train_miss_dir, "w")

    for line in file1:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            if "N" in seq:
                file2.write(seq + "\t" + label + "\n")

    file1.close()
    file2.close()


def create_sub_dataset(num_comp_seq, num_miss_seq, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, train_ratio):

    comp_sequences = []
    miss_sequences = []

    # randomly chosen sequences will be written into these two files
    file3 = open(train_subset_dir, "w")
    file4 = open(valid_subset_dir, "w")

    if num_comp_seq > 0:
        # All sequences in "train_comp_sequences.txt" are read and stored in the list
        # The sequences are read with their labels; line-by-line reading is done.
        # Hence, the labels do not need to be read separately
        file1 = open(train_comp_dir, "r")
        comp_sequences = file1.readlines()

        # random index values are sampled to choose the sequences in the list "comp_sequences" randomly
        # how many one of them will be chosen depends on the input argument "num_comp_seq"
        random_indices = np.random.choice(range(len(comp_sequences)), num_comp_seq)
        comp_sequences = [comp_sequences[i] for i in random_indices]

        file1.close()

    if num_miss_seq > 0:

        # All sequences in "train_missing_sequences.txt" are read and stored in the list
        # The sequences are read with their labels; line-by-line reading is done.
        # Hence, the labels do not need to be read separately
        file2 = open(train_miss_dir, "r")
        miss_sequences = file2.readlines()

        # random index values are sampled to choose the sequences in the list "miss_sequences" randomly
        # how many one of them will be chosen depends on the input argument "num_miss_seq"
        random_indices = np.random.choice(range(len(miss_sequences)), num_miss_seq)
        miss_sequences = [miss_sequences[i] for i in random_indices]

        file2.close()

    # n and k numbers of complete and missing sequences are chosen randomly above.
    # The lists of these sequences are combined and shuffled.
    sequences = comp_sequences + miss_sequences
    np.random.shuffle(sequences)

    # Shuffled sequence list is divided into train and test (validation).
    # Each line that has been read in if-statements contain both sequence and label
    # Hence, when they are split into train and test, labels would be also split.
    # The list [i for i in range(len(sequences))] is passed as argument so that the function can work.
    seq_train, seq_test, label_train, label_test = train_test_split(sequences, [i for i in range(len(sequences))],
                                                                    train_size=train_ratio)

    # train sequences are read into "train_subsequences.txt" file
    for seq in seq_train:
        file3.write(seq)
    # train sequences are read into "valid_subsequences.txt" file
    for seq in seq_test:
        file4.write(seq)

    file3.close()
    file4.close()


#complete_sequences(train_dir, train_comp_dir)
#missing_sequences(train_dir, dir3)
#create_sub_dataset(200_000, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 0.8)
