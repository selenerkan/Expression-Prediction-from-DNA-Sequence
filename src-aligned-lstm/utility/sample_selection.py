import pickle
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from autoencoder_dataset import PromoterDataset, collate_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SampleSelector():
    def __init__(self, samples_per_cluster, all_data_len, batch_size, n_clusters=8, root_dir="../../data", train_filename="train_sequences.txt") -> None:
        self.core_model = Autoencoder().to(device)
        self.core_model.load_state_dict(torch.load("../../src-Autoencoder/models-seed23/model-49.pth"))
        self.core_model.eval()

        with open("../../src-Autoencoder/cluster_model.pkl", "rb") as f:
            self.clustering_model = pickle.load(f)

        self.N_CLUSTERS = n_clusters
        self.all_data_len = all_data_len
        self.BATCH_SIZE = batch_size
        self.SAMPLES_PER_CLUSTER = samples_per_cluster

        train_set = PromoterDataset(root_dir, train_filename)
        self.data_loader = DataLoader(train_set, batch_size=self.BATCH_SIZE, collate_fn=collate_batch,shuffle=False)
        self.epoch_iterator = tqdm(
            self.data_loader, desc="Extracting latent space clusters", dynamic_ncols=True
        )

    def forward(self):
        all_clusters = []
        with torch.no_grad():
            for i, data in enumerate(self.epoch_iterator):
                seqs = data[0]
                outputs = self.core_model.encoder(seqs).flatten(start_dim=1)
                clusters = self.clustering_model.predict(outputs.cpu().numpy())
                all_clusters.append(clusters)
            all_clusters = np.concatenate(all_clusters)
        print("Selecting samples...")
        all_selections = []
        for c_id in range(self.N_CLUSTERS):
            # Randomly select samples from each cluster equally.
            try:
                selected_indices = np.random.choice(np.arange(self.all_data_len)[np.where(all_clusters==c_id)], 
                                        self.SAMPLES_PER_CLUSTER,replace=False) # p=distance to the closest test sample?
            except ValueError:
                # If there is not enough samples, take it all.
                selected_indices = np.arange(self.all_data_len)[np.where(all_clusters==c_id)]
            print(f"Cluster {c_id} size:",len(selected_indices))
            all_selections.append(selected_indices)

        all_selections = np.concatenate(all_selections)
        return all_selections