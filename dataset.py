import os
from typing import Tuple, List, Optional
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from sklearn.model_selection import KFold
from kipoiseq.transforms.functional import one_hot, fixed_len
import linecache as lc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, "datasets")

# You can use such a random seed 35813 (part of the Fibonacci Sequence).
np.random.seed(35813)


def transform(seq: str) -> np.ndarray:
    seq = seq[17:-13]  # remove primers
    padded_seq = fixed_len(seq, 112, "start", "X")
    one_hot_seq = one_hot(padded_seq, ["A", "C", "G", "T", "N"], "X", 0)
    return one_hot_seq


def complement_strand(strand: str) -> str:
    lookup_table = {"T": "A", "A": "T", "C": "G", "G": "C", "N": "N", "X": "X"}
    comp_nucs = []
    for nuc in strand:
        if nuc not in lookup_table.keys():
            comp_nuc = "X"
        else:
            comp_nuc = lookup_table[nuc]
        comp_nucs.append(comp_nuc)

    comp_strand = "".join(comp_nucs)
    return comp_strand


def batch_collate_fn(batch_data: List[Tuple[str, float]]) -> Tuple[Tensor, Tensor]:
    """Definition of how the batched data will be treated."""
    seq_list, label_list = [], []
    comp_seq_list = []

    for _sequence, _label in batch_data:
        seq_list.append(transform(_sequence))
        if _label is not None:
            label_list.append(np.array(_label, np.float32))
        else:
            label_list.append(np.array(np.nan))
        comp_seq_list.append(transform(complement_strand(_sequence)))

    sequences_arr = np.array(seq_list + comp_seq_list)
    labels_arr = np.array(label_list)

    sequences = torch.tensor(sequences_arr).permute(0, 2, 1).float()
    labels = torch.tensor(labels_arr).float()
    return sequences.to(device), labels.to(device)


class PromoterDataset(Dataset):
    """Base class for common functionalities of all datasets."""

    def __init__(
        self,
        path_to_data: str,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        in_memory: bool = False,
    ):
        """
        Dataset class to initialize data operations and preprocessing.

        Parameters
        ----------
        path_to_data: string
            Path where the expected data is stored.
        mode: string
            Which split to return, can be 'train', 'validation', 'test', or 'inference'.
            Use 'inference' to load all data if you have a pretrained model and want to use
            it in-production.
        n_folds: integer
            Number of cross validation folds.
        current_fold: integer
            Defines which cross validation fold will be selected for training.
        in_memory: bool
            Whether to store all data in memory or not.
        """
        super().__init__()
        assert (
            current_fold < n_folds
        ), "selected fold index cannot be more than number of folds."
        self.mode = mode
        self.n_folds = n_folds
        self.in_memory = in_memory
        self.path_to_data = path_to_data
        self.current_fold = current_fold

        if not mode == "inference" and in_memory:
            self.samples_labels = self.get_labels()

        self.n_samples_total = self.get_number_of_samples()

        # Keep half of the data as 'unseen' to be used in inference.
        self.seen_data_indices, self.unseen_data_indices = self.get_fold_indices(
            self.n_samples_total, 2, 0
        )

        if self.in_memory:
            self.loaded_samples = self.get_all_samples()

        if mode == "train" or mode == "validation":
            # Here split the 'seen' data to train and validation.
            if self.in_memory:
                self.seen_samples_labels = self.samples_labels[self.seen_data_indices]
                self.seen_samples_data = self.loaded_samples[self.seen_data_indices]

            self.n_samples_seen = len(self.seen_data_indices)
            self.tr_indices, self.val_indices = self.get_fold_indices(
                self.n_samples_seen,
                self.n_folds,
                self.current_fold,
            )

        if mode == "train":
            self.selected_indices = self.tr_indices
            if self.in_memory:
                self.samples_labels = self.seen_samples_labels[self.tr_indices]
                self.loaded_samples = self.seen_samples_data[self.tr_indices]
        elif mode == "validation":
            self.selected_indices = self.val_indices
            if self.in_memory:
                self.samples_labels = self.seen_samples_labels[self.val_indices]
                self.loaded_samples = self.seen_samples_data[self.val_indices]
        elif mode == "test":
            self.selected_indices = self.unseen_data_indices
            if self.in_memory:
                self.samples_labels = self.samples_labels[self.unseen_data_indices]
                self.loaded_samples = self.loaded_samples[self.unseen_data_indices]
        elif not mode == "inference":
            raise ValueError(
                "mode should be 'train', 'validation', 'test', or 'inference'"
            )
        if mode == "inference":
            self.n_samples_in_split = self.n_samples_total
        else:
            self.n_samples_in_split = len(self.selected_indices)

    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[int]]:
        if self.in_memory:
            if self.mode == "inference":
                label = None
            else:
                label = self.samples_labels[index]
            sequence = self.loaded_samples[index]
        else:
            sequence, label = self.get_sample_data(index)
        return self.preprocess(sequence), label

    def __len__(self) -> int:
        return self.n_samples_in_split

    def get_number_of_samples(self) -> int:
        """
        Method to find how many samples are expected in ALL dataset.
        E.g., number of images in the target folder, number of rows in dataframe.
        """
        with open(self.path_to_data, "r") as fp:
            length = len(fp.readlines())
        return length

    def get_labels(self) -> np.ndarray:
        """
        Method to read and store labels in a numpy array

        Returns
        -------
        labels: numpy ndarray
            An array stores the labels for each sample.
        """
        return pd.read_csv(self.path_to_data, sep="\t")["Expression"].values

    def get_sample_data(self, index: int) -> Tuple[str, str]:
        """
        If we cannot or do not want to store all the samples in memory, we need to
        read the data based on selected indices (train, validation or test).

        Parameters
        ----------
        index: integer
            In-split index of the expected sample.
            (e.g., 2 means the 3rd sample from validation split if mode is 'validation')

        Returns
        -------
        tensor: torch Tensor
            A torch tensor represents the data for the sample.
        """
        line = lc.getline(self.path_to_data, index + 1)
        seq, label = line.strip("\n").split("\t")
        return seq, label

    def preprocess(self, data: Tensor) -> Tensor:
        return data

    def get_all_samples(self) -> np.ndarray:
        """
        Convert data from all samples to the Torch Tensor objects to store in a list later.
        This function can be memory-consuming but time-saving, recommended to be used on small datasets.

        Returns
        -------
        all_data: np.ndarray
            A numpy array represents all data.
        """
        return pd.read_csv(self.path_to_data, sep="\t")["Sequence"].values

    def get_fold_indices(
        self, all_data_size: int, n_folds: int, fold_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create folds and get indices of train and validation datasets.

        Parameters
        --------
        all_data_size: int
            Size of all data.
        fold_id: int
            Which cross validation fold to get the indices for.

        Returns
        --------
        train_indices: numpy ndarray
            Indices to get the training dataset.
        val_indices: numpy ndarray
            Indices to get the validation dataset.
        """
        kf = KFold(n_splits=n_folds, shuffle=True)
        split_indices = kf.split(range(all_data_size))
        train_indices, val_indices = [
            (np.array(train), np.array(val)) for train, val in split_indices
        ][fold_id]
        # Split train and test
        return train_indices, val_indices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} dataset ({self.mode}) with"
            f" n_samples={self.n_samples_in_split}, "
            f"current fold:{self.current_fold+1}/{self.n_folds}"
        )
