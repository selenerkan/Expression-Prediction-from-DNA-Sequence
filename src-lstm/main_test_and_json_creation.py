
import torch
import json
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

from crnn_model2 import PromoterNet
from dataset import PromoterDataset, transform, complement_strand

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("TEST DEVICE:",device)

np.random.seed(23)
torch.manual_seed(23)

# *********************************************************************************************************

def collate_batch_test(batch):

    seq_list, label_list = [], []
    comp_seq_list, comp_label_list = [], []

    for (_sequence, _label) in batch:
        seq_list.append(transform(_sequence))
        label_list.append(np.array(0, np.float32))
        comp_seq_list.append(transform(complement_strand(_sequence)))

    sequences = np.array(seq_list + comp_seq_list)
    labels = np.array(label_list)

    sequences = torch.tensor(sequences).permute(0, 2, 1).float()
    labels = torch.tensor(labels).float()
    return sequences.to(device), labels.to(device)
        
root_dir = "../data"
test_filename = "test_sequences.txt"
test_set = PromoterDataset(root_dir, test_filename)
test_loader = DataLoader(test_set, batch_size=1024, collate_fn=collate_batch_test)

with torch.no_grad():
    model = PromoterNet()
    model = model.to(device)
    PATH = './models-seed23-cluster_balanced/model4-47.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()

    preds = []
    for i, data in enumerate(test_loader):
        seqs, labels = data[0], data[1]
        outputs = model(seqs)
        preds.append(outputs)
        
preds = torch.cat(preds,dim=0).cpu().numpy()
import matplotlib.pyplot as plt

# matplotlib histogram
plt.hist(preds, color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.show()
print(preds.shape)

with open('../data/sample_submission.json', 'r') as f:
    ground = json.load(f)
    
indices = np.array([int(indice) for indice in list(ground.keys())])
PRED_DATA = OrderedDict()
for i in indices:
    PRED_DATA[str(i)] = float(preds[i])
    
with open('pred-lstm-cluster_balanced.json', 'w') as f:
    json.dump(PRED_DATA,f)
