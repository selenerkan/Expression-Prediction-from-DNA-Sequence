import linecache
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

import pickle
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

def sample_selection():
    root_dir = "../data"
    train_filename = "train_subsequences_half.txt"

    train_set = PromoterDataset(root_dir, train_filename)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_batch,shuffle=False)

    model = Autoencoder()
    model = model.to(device)

    model.load_state_dict(torch.load("./models-seed23/model-49.pth"))

    model.eval()
    latent_list = []
    
    #labels = []
    #all_seqs = []
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
            #latent_list.append(outputs.cpu())
            cluster_model = cluster_model.partial_fit(outputs.cpu().numpy())
        #latent_list = torch.cat(latent_list,dim=0).numpy()
    # Cluster the latent space.
    print("Samples clustered !")
    with open("cluster_model.pkl", "wb") as f:
        pickle.dump(cluster_model, f)
    # with open("cluster_model.pkl", "rb") as f:
    #     loaded_model = pickle.load(f)

    # clusters = loaded_model.predict(latent_list)

    # clusters = loaded_model.predict(latent_list)
    # Collect samples.
    # all_selected = []
    # for c_id in range(N_CLUSTERS):
    #     # Randomly select samples from each cluster equally.
    #     indices_from_cluster = np.random.choice(np.arange(MAX_SAMPLES)[np.where(labels==c_id)], SAMPLES_PER_CLUSTER) # p=distance to the closest test sample?
    #     #all_selected.append(latent_list[indices_from_cluster])
    #     all_selected.append(indices_from_cluster)
    # all_selected = np.array(all_selected)
    # # Concat sample sets selected for each cluster.
    # #all_selected=all_selected.reshape(all_selected.shape[0]*all_selected.shape[1],all_selected.shape[2])
    # np.save("selected_sample_indices.npy",all_selected)

    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(latent_list)
    # plt.scatter(X_embedded[:,0],X_embedded[:,1],c=clusters,label=clusters)
    # plt.show()

sample_selection()
