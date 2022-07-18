import linecache
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchmetrics.functional as metrics_F

from torch.utils.data import DataLoader
from torchsummary import summary

from crnn_model import PromoterNet
from dataset import PromoterDataset, transform, complement_strand
from utility.early_stopping import EarlyStopping
from utility.choose_subsequences import complete_sequences, missing_sequences, create_sub_dataset

import json
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(23)
torch.manual_seed(23)
'''
Requirments: 

../data/sample_submission.json - sample_submission.json can be downloaded from challenge repository in git.
../data/test_sequences.txt
Enter PATH to the saved model in line 123 that is to be run on the test data.


'''



# *************************************** DATA CREATION **************************************************
train_dir = "../data/train_sequences.txt"
train_comp_dir = "../data/train_comp_sequences_seed23.txt"
train_miss_dir = "../data/train_missing_sequences_seed23.txt"
train_subset_dir = "../data/train_subsequences_seed23.txt"
valid_subset_dir = "../data/valid_subsequences_seed23.txt"


#complete_sequences(train_dir, train_comp_dir)
#missing_sequences(train_dir, train_miss_dir)
#create_sub_dataset(400_000, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 0.9)

# *********************************************************************************************************


def collate_batch_test(batch):

    seq_list, label_list = [], []
    comp_seq_list, comp_label_list = [], []

    for (_sequence, _label) in batch:
        seq_list.append(transform_test(_sequence))
        label_list.append(np.array(0, np.float32))
        comp_seq_list.append(transform_test(complement_strand_test(_sequence)))

    sequences = np.array(seq_list + comp_seq_list)
    labels = np.array(label_list)

    sequences = torch.tensor(sequences).permute(0, 2, 1).float()
    labels = torch.tensor(labels).float()
    return sequences.to(device), labels.to(device)

        
root_dir = "../data"
test_filename = "test_sequences.txt"


#train_dataset = PromoterSeqDataset(root_dir, filename, transforms)
#loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)
test_set = PromoterDataset(root_dir, test_filename)


test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_batch_test)

with torch.no_grad():
    model = PromoterNet(cnn_dropout=0.2)
    model = model.to(device)
    PATH = 'models-seed23-dropout/model4-19.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()

    Y = []
    for i, data in enumerate(test_loader):

        seqs, labels = data[0], data[1]
        outputs = model(seqs)
        Y.append(outputs)
        
Y_pred = []
for i in Y:
    Y_pred.append(float(i))
Y_pred = np.array(Y_pred)


with open('../data/sample_submission.json', 'r') as f:
    ground = json.load(f)
    
indices = np.array([int(indice) for indice in list(ground.keys())])
PRED_DATA = OrderedDict()
for i in indices:
    PRED_DATA[str(i)] = float(Y_pred[i])
    
def dump_predictions(prediction_dict, prediction_file):
    with open(prediction_file, 'w') as f:
        json.dump(prediction_dict,f)
dump_predictions(PRED_DATA,'pred.json')


