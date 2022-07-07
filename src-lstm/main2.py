import linecache
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchmetrics.functional as metrics_F

from torch.utils.data import DataLoader
from torchsummary import summary

from crnn_model import PromoterNet
from dataset import PromoterDataset, collate_batch
from utility.early_stopping import EarlyStopping
from utility.choose_subsequences import complete_sequences, missing_sequences, create_sub_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(23)
torch.manual_seed(23)

# *************************************** DATA CREATION **************************************************
train_dir = "../data/train_sequences.txt"
train_comp_dir = "../data/train_comp_sequences_seed23.txt"
train_miss_dir = "../data/train_missing_sequences_seed23.txt"
train_subset_dir = "../data/train_subsequences_seed23.txt"
valid_subset_dir = "../data/valid_subsequences_seed23.txt"

complete_sequences(train_dir, train_comp_dir)
missing_sequences(train_dir, train_miss_dir)
create_sub_dataset(400_000, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 0.9)

# *********************************************************************************************************
n_max_epochs = 20
train_loss_list, valid_loss_list = [], []
train_r2_list, valid_r2_list = [], []

def training_loop(n_max_epochs):
    root_dir = "../data"
    train_filename = "train_subsequences_seed23.txt"
    valid_filename = "valid_subsequences_seed23.txt"

    #train_dataset = PromoterSeqDataset(root_dir, filename, transforms)
    #loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)

    train_set = PromoterDataset(root_dir, train_filename)
    valid_set = PromoterDataset(root_dir, valid_filename)

    train_loader = DataLoader(train_set, batch_size=1024, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_set, batch_size=1024, collate_fn=collate_batch)

    model = PromoterNet(cnn_dropout=0.2)
    model = model.to(device)
    # summary(model, input_size=(5, 112))

    loss_fn = torch.nn.MSELoss()
    r2_metric_fn = metrics_F.r2_score
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

    file = open("./train_results.txt", "w+")
    early_stopping = EarlyStopping(patience=8)

    for epoch in range(n_max_epochs):

        model.train()
        train_loss, train_r2 = 0.0, 0.0
        valid_loss, valid_r2 = 0.0, 0.0

        print()
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

        #del loss, outputs
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_loader):

                seqs, labels = data[0], data[1]
                outputs = model(seqs)
                loss = loss_fn(outputs, labels)
                r2_value = r2_metric_fn(outputs, labels)

                valid_loss += loss#.detach()
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
        #del loss, outputs
        save_dir = "./models-seed23-dropout"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file_name = "model4-" + str(epoch) + ".pth"
        torch.save(model.state_dict(), os.path.join(save_dir, save_file_name))
        linecache.clearcache()

def plot():
    plt.plot(range(1, len(train_loss_list)), train_loss_list, color="blue", label="training_loss")
    plt.plot(range(1, len(valid_loss_list)), valid_loss_list, color="red", label="validation_loss")
    plt.legend()
    plt.show()

    plt.plot(range(1, len(train_r2_list)), train_r2_list, color="blue", label="training_r2")
    plt.plot(range(1, len(valid_r2_list)), valid_r2_list, color="red", label="validation_r2")
    plt.legend()
    plt.show()

try:
    epoch = training_loop(n_max_epochs)
except KeyboardInterrupt:
    plot()
