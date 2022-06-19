
from src.model import *
from src.dataset import *
from torchsummary import summary

root_dir = "/Users/goktug/PycharmProjects/ML4RG Project/data"
filename = "train_sequences.txt"
train_dataset = PromoterSeqDataset(root_dir, filename, transforms)
loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)

net = PromoterNet()
summary(net, input_size=(5, 112))
# for samples, labels in loader:
