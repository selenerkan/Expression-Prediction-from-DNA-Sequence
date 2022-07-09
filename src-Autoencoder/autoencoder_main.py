import linecache
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchmetrics.functional as metrics_F

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchsummary import summary

from autoencoder import Autoencoder
from autoencoder_dataset import PromoterDataset, collate_batch
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

    train_set = PromoterDataset(root_dir, train_filename)
    valid_set = PromoterDataset(root_dir, valid_filename)

    train_loader = DataLoader(train_set, batch_size=1024, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_set, batch_size=1024, collate_fn=collate_batch)

    model = Autoencoder()
    model = model.to(device)
    # summary(model, input_size=(5, 112))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

    file = open("./train_results.txt", "w+")
    early_stopping = EarlyStopping(patience=8)

    for epoch in range(n_max_epochs):

        model.train()
        total_train_loss = 0.0

        print()
        for i, data in enumerate(train_loader):

            seqs, recons_seqs = data[0], data[1]
            optimizer.zero_grad()

            predicted_seqs = model(seqs)  # model reconstruct the sequences by encoder-decoder
            loss = loss_fn(predicted_seqs, recons_seqs)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.detach()

        ave_train_loss = total_train_loss / len(train_loader)
        train_loss_list.append(ave_train_loss.cpu())

        print(f"Train Epoch {epoch + 1} Loss: {ave_train_loss:.4f}")
        file.write(f"Train Epoch {epoch + 1} Loss: {ave_train_loss:.4f}")

        linecache.clearcache()

    model.eval()
    latent_list = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            seqs, recons_seqs = data[0], data[1]
            outputs = model.encoder(seqs)
            latent_list.append(outputs)

        latent_list = torch.cat(latent_list).cpu().numpy()
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(latent_list)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

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