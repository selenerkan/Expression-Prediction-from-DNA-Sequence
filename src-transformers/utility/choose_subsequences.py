from sklearn.model_selection import train_test_split
import numpy as np
from Bio.GenBank import FeatureParser
from Bio.Align import PairwiseAligner
from tqdm import tqdm

def sequence_features(train_dir):

    with open("../../data/yeast pTpA GPRA vector with N80.gb") as file_handle:
        parser = FeatureParser()
        seq_record = parser.parse(file_handle)

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = -3
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -2
    aligner.query_right_open_gap_score = 0
    aligner.query_left_open_gap_score = 0
    aligner.query_right_extend_gap_score = 0
    aligner.query_left_extend_gap_score = 0


    file1 = open(train_dir, "r")
    tr_data = file1.readlines()

    all_features,all_complement_needs=[],[]
    for idx in tqdm(range(len(tr_data))):
        line = tr_data[idx]
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]

            alignments = aligner.align(seq_record.seq,seq[17:-13])
            al_start = alignments[0].aligned[0][0][0]
            al_end = alignments[0].aligned[0][-1][-1]

            seq_features_one_hot = np.zeros(len(seq_record.features),dtype=np.int32)
            need_complement = np.zeros(len(seq_record.features),dtype=np.int32)
            for i,feature in enumerate(seq_record.features):
                if int(feature.location.start) > al_end or int(feature.location.end) < al_start:
                    # No overlap
                    continue
                else:
                    if feature.strand == -1:
                        need_complement[i] = 1
                    seq_features_one_hot[i] = 1
            all_features.append(seq_features_one_hot)
            all_complement_needs.append(need_complement)
    all_features = np.array(all_features)
    all_complement_needs = np.array(all_complement_needs)
    np.save("../../data/test_features.npy",all_features)
    np.save("../../data/test_complement_needs.npy",all_complement_needs)

def clustered_sequences(train_dir, train_clustered_dir,samples_per_cluster=361, batch_size=20000):
    from sample_selection import SampleSelector
    file1 = open(train_dir, "r")
    all_data = file1.readlines()
    all_data_len = len(all_data)
    #file1.seek(0)
    file2 = open(train_clustered_dir, "w")

    selector = SampleSelector(samples_per_cluster=samples_per_cluster, all_data_len=all_data_len, batch_size=batch_size,train_filename=train_dir.split("/")[-1])
    selected_indices = selector.forward()

    for idx in selected_indices:
        line = all_data[idx]
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            file2.write(seq + "\t" + label + "\n")
    file1.close()
    file2.close()

def floating_sequences(train_dir, train_float_dir):

    file1 = open(train_dir, "r")
    file2 = open(train_float_dir, "w")

    for line in file1:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            if int(float(label)) != float(label):
                file2.write(seq + "\t" + label + "\n")

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


def choose_categorized_sequences2(cat_sequences, num_requested_seq):

    temp = []
    num_categories = 0
    for category in cat_sequences:
        if len(category) != 0:
            temp.append(category)
            num_categories += 1

    cat_sequences = temp

    remainder = num_requested_seq % num_categories
    division = num_requested_seq // num_categories  # average number of sequences that will be chosen from each category
    all_sequences = []

    for category in cat_sequences:
        replace = True if len(category) < division else False  # replacing for only the categories with sequences less than ave.
        random_indices = np.random.choice(range(len(category)), division, replace=replace)
        chosen_sequences = [category[j] for j in random_indices]
        all_sequences += chosen_sequences

    random_indices = np.random.choice(range(len(cat_sequences[0])), remainder, replace=False)
    chosen_sequences = [cat_sequences[0][j] for j in random_indices]
    all_sequences += chosen_sequences

    return all_sequences

def create_sub_dataset(num_comp_seq, num_miss_seq, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, train_ratio, auto_balance=True):

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
        if auto_balance:
            cat_sequences = read_and_split_sequences(file1)
            comp_sequences = choose_categorized_sequences(cat_sequences, num_comp_seq)
        else:
            comp_sequences = file1.readlines()
            random_indices = np.random.choice(range(len(comp_sequences)), num_comp_seq)
            comp_sequences = [comp_sequences[i] for i in random_indices]

        file1.close()

    if num_miss_seq > 0:

        # All sequences in "train_missing_sequences.txt" are read and stored in the list
        # The sequences are read with their labels; line-by-line reading is done.
        # Hence, the labels do not need to be read separately
        file2 = open(train_miss_dir, "r")
        if auto_balance:
            cat_sequences = read_and_split_sequences(file2)
            miss_sequences = choose_categorized_sequences(cat_sequences, num_miss_seq)
        else:
            miss_sequences = file2.readlines()
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

if __name__=="__main__":
    # train_dir = "../../data/train_sequences.txt"
    # train_clustered_dir = "../../data/train_cluster_sequences361samples.txt"
    # clustered_sequences(train_dir, train_clustered_dir)

    train_dir = "../../data/train_sequences.txt"
    train_clustered_dir = "../../data/train_cluster_100ksequences.txt"
    
    clustered_sequences(train_dir, train_clustered_dir, samples_per_cluster=100000)
    #sequence_features(train_dir)

    # root_dir = "../../data"

    # train_dir = root_dir + "/train_sequences.txt"
    # train_comp_dir = root_dir + "/train_comp_sequences_balanced.txt"
    # train_miss_dir = root_dir + "/train_missing_sequences_balanced.txt"
    # train_subset_dir = root_dir + "/train_subsequences_balanced.txt"
    # valid_subset_dir = root_dir + "/valid_subsequences_balanced.txt"

    # # open(train_dir, 'a').close()
    # # open(train_comp_dir, 'a').close()
    # # open(train_miss_dir, 'a').close()
    # # open(train_subset_dir, 'a').close()
    # # open(valid_subset_dir, 'a').close()

    # complete_sequences(train_dir, train_comp_dir)
    # missing_sequences(train_dir, train_miss_dir)
    # create_sub_dataset(100_000, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 0.8)
