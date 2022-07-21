import numpy as np
import torch

import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from utility.choose_subsequences import complete_sequences, missing_sequences, create_sub_dataset
from autoencoder_dataset import PromoterDataset, collate_batch

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(23)
torch.manual_seed(23)

MAX_SAMPLES = 6_739_258
MAX_SAMPLES = 5000
SAMPLES_PER_CLUSTER = 1000
N_CLUSTERS = 8
BATCH_SIZE=1024

# *************************************** DATA CREATION **************************************************
train_dir = "../data/train_sequences.txt"
train_comp_dir = "../data/train_comp_sequences_half.txt"
train_miss_dir = "../data/train_missing_sequences_half.txt"

train_subset_dir = "../data/train_subsequences_half.txt"

complete_sequences(train_dir, train_comp_dir)
missing_sequences(train_dir, train_miss_dir)
create_sub_dataset(MAX_SAMPLES, 0, train_comp_dir, train_miss_dir, train_subset_dir)

# *********************************************************************************************************

def train_clustering():
    root_dir = "../data"
    train_filename = "train_subsequences_half.txt"

    train_set = PromoterDataset(root_dir, train_filename)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_batch,shuffle=False)

    model = Autoencoder()
    model = model.to(device)

    model.load_state_dict(torch.load("./models-seed23/model-49.pth"))

    model.eval()

    cluster_model = MiniBatchKMeans(n_clusters=N_CLUSTERS,max_iter=500,batch_size=BATCH_SIZE,)
    epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
    with torch.no_grad():
        for i, data in enumerate(epoch_iterator):
            epoch_iterator.set_description(
                "Extracting latent space clusters (%d / %d Steps)"
                % (i, MAX_SAMPLES//BATCH_SIZE)
            )
            seqs = data[0]
            outputs = model.encoder(seqs).flatten(start_dim=1)
            cluster_model = cluster_model.partial_fit(outputs.cpu().numpy())
    # Cluster the latent space.
    print("Samples clustered !")
    with open("cluster_model.pkl", "wb") as f:
        pickle.dump(cluster_model, f)

    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(latent_list)
    # plt.scatter(X_embedded[:,0],X_embedded[:,1],c=clusters,label=clusters)
    # plt.show()

train_clustering()
