

def find_complete_seq():
    counter = 0
    index, max_len = 0, 0

    all_sequences = []
    train_file = open("/Users/goktug/PycharmProjects/ML4RG Project/data/train_sequences.txt", "r")
    train_comp_file = open("/Users/goktug/PycharmProjects/ML4RG Project/data/train_comp_sequences.txt", "w")

    for line in train_file:
        sample = line.strip("\n").split("\t")
        if len(sample) == 2:
            seq, label = sample[0], sample[1]
            if "N" not in seq:
                train_comp_file.write(seq + "\t" + label + "\n")

    train_file.close()
    train_comp_file.close()


find_complete_seq()