import linecache
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from utility.choose_subsequences import complete_sequences, missing_sequences, create_sub_dataset
from autoencoder_dataset import PromoterDataset, collate_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(23)
torch.manual_seed(23)

# *************************************** DATA CREATION **************************************************
train_dir = "../data/train_sequences.txt"
train_comp_dir = "../data/train_comp_sequences_half.txt"
train_miss_dir = "../data/train_missing_sequences_half.txt"
train_subset_dir = "../data/train_subsequences_half.txt"
valid_subset_dir = "../data/valid_subsequences_half.txt"

complete_sequences(train_dir, train_comp_dir)
missing_sequences(train_dir, train_miss_dir)
create_sub_dataset(10_000, 0, train_comp_dir, train_miss_dir, train_subset_dir, valid_subset_dir, 1.0)

# *********************************************************************************************************
n_max_epochs = 50
train_loss_list, valid_loss_list = [], []
train_r2_list, valid_r2_list = [], []


def training_loop():
    root_dir = "../data"
    train_filename = "train_subsequences_half.txt"

    train_set = PromoterDataset(root_dir, train_filename)

    train_loader = DataLoader(train_set, batch_size=1024, collate_fn=collate_batch)

    model = Autoencoder()
    model = model.to(device)

    model.load_state_dict(torch.load("./models-seed23/model-49.pth"))

    model.eval()
    latent_list = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            seqs, _, label = data[0], data[1], data[2]
            outputs = model.encoder(seqs).flatten(start_dim=1)
            latent_list.append(outputs.cpu())
            labels.append(label.cpu())

        latent_list = torch.cat(latent_list).numpy()
        labels = torch.cat(labels).numpy()

    print(latent_list.shape)
    print(labels.shape)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(latent_list)
    # for i in range(18):
    #     X_indices = np.where(np.floor(labels)==i)
    #     plt.scatter(X_embedded[X_indices, 0], X_embedded[X_indices, 1],label=labels[X_indices],c=labels[X_indices], cmap='Blues', edgecolors="black")
    plt.scatter(X_embedded[:,0],X_embedded[:,1],label=labels,c=labels,cmap="Blues",edgecolors="black")
    plt.colorbar()
    plt.show()

def plot():
    plt.plot(range(1, len(train_loss_list)), train_loss_list, color="blue", label="training_loss")
    plt.plot(range(1, len(valid_loss_list)), valid_loss_list, color="red", label="validation_loss")
    plt.legend()
    plt.show()

    plt.plot(range(1, len(train_r2_list)), train_r2_list, color="blue", label="training_r2")
    plt.plot(range(1, len(valid_r2_list)), valid_r2_list, color="red", label="validation_r2")
    plt.legend()
    plt.show()

epoch = training_loop()
