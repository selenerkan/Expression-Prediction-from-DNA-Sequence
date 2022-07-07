from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(0)

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


def read_and_split_sequences(file_ptr):

    sequences = [[] for i in range(18)]

    for line in file_ptr:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            index = int(float(label))
            sequences[index].append(line)

    return sequences


def choose_categorized_sequences(cat_sequences, num_requested_seq):

    additional = 0
    num_categories = len(cat_sequences)
    remainder = num_requested_seq % num_categories
    division = num_requested_seq // num_categories  # average number of sequences that will be chosen from each category
    all_sequences = []

    # we have 18 categories, and each category has some number of sequences
    # The number of sequences contained by each category is stored in "size_of_categories"
    # How many number of sequences will be chosen from each category will be stored in "num_chosen_seq".
    size_of_categories = [len(category) for category in cat_sequences]
    num_chosen_seq = [0 for i in range(len(cat_sequences))]

    for index in range(num_categories):
        if size_of_categories[index] < division:  # the size of category is lower than average number of sequences
            additional += (division - size_of_categories[index])  # the difference is summed in a variable "additional"
            num_chosen_seq[index] = size_of_categories[index]  # for that category, its maximum size of sequences is chosen
        else:
            num_chosen_seq[index] = division # the size of category is equal or greater than average number of sequences

    large_categories = [[i, 0] for i, size in enumerate(size_of_categories) if (size - division) >= (additional + remainder)]
    division2 = (additional + remainder) // len(large_categories)
    remainder2 = (additional + remainder) % len(large_categories)
    large_categories = [[item[0], division2] for item in large_categories]
    large_categories[-1][1] += remainder2

    for category in large_categories:
        index, dist = category
        num_chosen_seq[index] += dist

    for i, num in enumerate(num_chosen_seq):
        one_category = cat_sequences[i]
        random_indices = np.random.choice(range(len(one_category)), num)
        chosen_sequences = [one_category[j] for j in random_indices]
        all_sequences += chosen_sequences

    return all_sequences


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
        cat_sequences = read_and_split_sequences(file1)
        comp_sequences = choose_categorized_sequences(cat_sequences, num_comp_seq)

        file1.close()

    if num_miss_seq > 0:

        # All sequences in "train_missing_sequences.txt" are read and stored in the list
        # The sequences are read with their labels; line-by-line reading is done.
        # Hence, the labels do not need to be read separately
        file2 = open(train_miss_dir, "r")
        cat_sequences = read_and_split_sequences(file2)
        miss_sequences = choose_categorized_sequences(cat_sequences, num_miss_seq)

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

if __name__=="__main__":

    root_dir = "../../data"

    train_dir = root_dir + "/train_sequences.txt"
    train_comp_dir = root_dir + "/train_comp_sequences_balanced.txt"
    train_miss_dir = root_dir + "/train_missing_sequences_balanced.txt"
    train_subset_dir = root_dir + "/train_subsequences_balanced.txt"
    valid_subset_dir = root_dir + "/valid_subsequences_balanced.txt"

    # open(train_dir, 'a').close()
    # open(train_comp_dir, 'a').close()
    # open(train_miss_dir, 'a').close()
    # open(train_subset_dir, 'a').close()
    # open(valid_subset_dir, 'a').close()

    complete_sequences(train_dir, train_comp_dir)
    missing_sequences(train_dir, train_miss_dir)
    create_sub_dataset(100_000, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 0.8)
