from dataset import *

from utility.early_stopping import *
from utility.choose_subsequences import *


root_dir = "./data"
train_filename = "train_subsequences.txt"
valid_filename = "valid_subsequences.txt"

# *************************************** DATA CREATION **************************************************
# train_dir = root_dir + "/train_sequences.txt"
# train_comp_dir = root_dir + "/train_comp_sequences.txt"
# train_miss_dir = root_dir + "/train_missing_sequences.txt"
# train_subset_dir = root_dir + "/train_subsequences.txt"
# valid_subset_dir = root_dir + "/valid_subsequences.txt"

# complete_sequences(train_dir, train_comp_dir)
# missing_sequences(train_dir, train_miss_dir)
# create_sub_dataset(1_000_000, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 0.8)

# *********************************************************************************************************

train_set = PromoterDataset(root_dir, train_filename)
valid_set = PromoterDataset(root_dir, valid_filename)


train_loader = DataLoader(train_set, batch_size=1024, collate_fn=collate_batch_transformers)
valid_loader = DataLoader(valid_set, batch_size=1024, collate_fn=collate_batch_transformers)


for i, data in enumerate(train_loader):

    seqs, labels = data[0], data[1]
    print(seqs)
    break
