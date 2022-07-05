import torch
import os
import linecache
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utility.early_stopping import EarlyStopping
from utility.choose_subsequences import complete_sequences, missing_sequences, create_sub_dataset
from torchsummary import summary
import torchmetrics.functional as metrics_F

from transformer_model import ConvTransformer, TransformerNet
from dataset import PromoterDataset, collate_batch, GetLabels

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUNS ON:",device)

# *************************************** PARAMETERS **************************************************

MAX_EPOCHS = 50
BATCH_SIZE = 1024
MAX_SEQ = 1_000_000
LR = 0.001
REG = 0.001

root_dir = "../data"
train_filename = "train_subsequences.txt"
valid_filename = "valid_subsequences.txt"

# *************************************** DATA CREATION **************************************************
train_dir = root_dir + "/train_sequences.txt"
train_comp_dir = root_dir + "/train_comp_sequences.txt"
train_miss_dir = root_dir + "/train_missing_sequences.txt"
train_subset_dir = root_dir + "/train_subsequences.txt"
valid_subset_dir = root_dir + "/valid_subsequences.txt"

complete_sequences(train_dir, train_comp_dir)
missing_sequences(train_dir, train_miss_dir)
create_sub_dataset(MAX_SEQ, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 0.8)

# *********************************************************************************************************
classes = GetLabels(num_classes = 500)
train_set = PromoterDataset(root_dir, train_filename, classes)
valid_set = PromoterDataset(root_dir, valid_filename, classes)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_batch)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, collate_fn=collate_batch)

# uses positional enoders from the postional encoder class
# implement this inside the transformers class before calling the model
model = ConvTransformer()
model = model.to(device)
summary(model, input_size=(5, 112))

loss_fn = torch.nn.MSELoss()
r2_metric_fn = metrics_F.r2_score
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)

file = open("../results/train_results_transformers.txt", "w")
early_stopping = EarlyStopping(patience=8)
train_loss_list, valid_loss_list = [], []
train_r2_list, valid_r2_list = [], []

# *************************************************** TRAINING LOOP ******************************************************
for epoch in range(MAX_EPOCHS):

    model.train()
    train_loss, train_r2 = 0.0, 0.0
    valid_loss, valid_r2 = 0.0, 0.0

    for i, data in enumerate(train_loader):
        seqs, labels = data[0], data[1]
        optimizer.zero_grad()

        outputs = model(seqs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.detach()

        r2_value = r2_metric_fn(outputs.cpu().detach(), labels.cpu().detach())
        train_r2 += r2_value.item()

    ave_train_loss = train_loss / len(train_loader)
    ave_train_r2 = train_r2 / len(train_loader)
    train_loss_list.append(ave_train_loss.cpu())
    train_r2_list.append(ave_train_r2)

    print(f"Train Epoch {epoch + 1} Loss: {ave_train_loss:.4f}, R2-Score: {ave_train_r2:.4f}")
    file.write(f"Train Epoch {epoch + 1} Loss: {ave_train_loss:.4f}, R2-Score: {ave_train_r2:.4f}\n")

# *************************************************** VALIDATION ******************************************************

    # del loss, outputs
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            seqs, labels = data[0], data[1]
            outputs = model(seqs)
            loss = loss_fn(outputs, labels)
            r2_value = r2_metric_fn(outputs, labels)

            valid_loss += loss  # .detach()
            valid_r2 += r2_value

        average_loss = valid_loss / len(valid_loader)
        early_stopping.step(average_loss)
        if early_stopping.check_patience():
            break

        ave_valid_loss = valid_loss / len(valid_loader)
        ave_valid_r2 = valid_r2 / len(valid_loader)
        valid_loss_list.append(ave_valid_loss.cpu())
        valid_r2_list.append(ave_valid_r2.cpu())

        print(f"Validation Epoch {epoch + 1} Loss: {ave_valid_loss:.4f}, R2-Score: {ave_valid_r2:.4f}")
        file.write(f"Validation Epoch {epoch + 1} Loss: {ave_valid_loss:.4f}, R2-Score: {ave_valid_r2:.4f}\n")
    # del loss, outputs
    save_dir = "../models-transformers"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_name = "model4-" + str(epoch) + ".pth"
    torch.save(model.state_dict(), os.path.join(save_dir, save_file_name))
    linecache.clearcache()

# *************************************************** COLLECTING RESULTS ******************************************************

epochs = range(1, MAX_EPOCHS + 1)
plt.plot(epochs, train_loss_list, color="blue", label="training_loss")
plt.plot(epochs, valid_loss_list, color="red", label="validation_loss")
plt.legend()
plt.show()

plt.plot(epochs, train_r2_list, color="blue", label="training_r2")
plt.plot(epochs, valid_r2_list, color="red", label="validation_r2")
plt.legend()
plt.show()
