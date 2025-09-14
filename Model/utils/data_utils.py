import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx
import json
from box import Box
import yaml
from typing import Optional
from pathlib import Path
from typing import Tuple
import random


CONFIG_FILE = "./Model/config.yaml"
GRAPH_FILE = "graph.npy"
META_FILE = "meta.json"


def get_graph(dataset_path):
    """
    Load the graph topology from the graph.npy file.
    """
    path = os.path.join(dataset_path, GRAPH_FILE)
    edge_index = np.load(path)

    edge_list = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_edges_from(edge_list)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def get_meta_data(dataset_path):
    """
    Load the metadata from the meta.json file.
    """
    path = os.path.join(dataset_path, META_FILE)
    with open(path, "r") as f:
        metadata = json.load(f)

    return metadata


def get_config(config_path: Optional[str] = None) -> Box:
    """
    Load the configuration from a provided path or fall back to the default.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, CONFIG_FILE)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    config = Box(cfg)
    return config


def load_dataset(dataset_path):
    """
    Load all case simulations for the selected network.
    """
    print("Loading dataset...")

    case_dir = os.path.join(dataset_path, "cases")
    X, Y = [], []

    for filename in sorted(os.listdir(case_dir)):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(case_dir, filename))
            X.append(data["x"])
            Y.append(data["y"])

    X = np.stack(X)  # shape: [num_cases, timesteps, num_buses]
    Y = np.stack(Y)  # shape: [num_cases, num_lines]

    print("Dataset loaded.")

    return X, Y


# regular data split
def get_data_loaders(
        dataset_path,
        batch_size,
        *,
        test_size = 0.2,
        val_size = 0.1,
        cal_size = 0.2,
        seed = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, tuple]:
    """
    Load the dataset and create data loaders for training, validation,
    calibration and testing.

    Parameters
    ----------
    dataset_path : str | Path
        Folder that contains the raw *.npy / *.pt files.
    batch_size : int
        Mini-batch size for every DataLoader.
    test_size : float, default=0.2
        Fraction (0–1) of the full dataset used **only once** at the very end.
    val_size : float, default=0.1
        Fraction used for hyper-parameter tuning / early stopping.
    cal_size : float, default=0.2
        Fraction reserved for any *post-training* calibration (e.g., temperature-scaling).
    seed : int, default=42
        Random seed for reproducible splits.

    Returns
    -------
    train_loader, val_loader, cal_loader, test_loader, input_shape
    """

    # --------------------------- load ----------------------------
    X, Y = load_dataset(dataset_path)

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    Y_tensor = torch.as_tensor(Y, dtype=torch.float32)

    # Ensure (B, T, K, N) shape
    if X_tensor.dim() == 3:                          # (B, T, N)
        X_tensor = X_tensor.unsqueeze(2)             # (B, T, 1, N)
    elif X_tensor.dim() != 4:
        raise ValueError(f"Unsupported X shape: {X_tensor.shape}")

    # Label normalisation  – each row sums to 1
    Y_sum = Y_tensor.sum(dim=1, keepdim=True).clamp(min=1.0)
    Y_tensor = Y_tensor / Y_sum

    dataset = TensorDataset(X_tensor, Y_tensor)
    num_samples = len(dataset)

    # --------------------------- split ---------------------------
    if (test_size + val_size + cal_size) >= 1.0:
        raise ValueError("test_size + val_size + cal_size must be < 1.0")

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=g)

    n_test = int(num_samples * test_size)
    n_val  = int(num_samples * val_size)
    n_cal  = int(num_samples * cal_size)
    n_train = num_samples - n_test - n_val - n_cal

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    cal_idx   = indices[n_train + n_val:n_train + n_val + n_cal]
    test_idx  = indices[n_train + n_val + n_cal:]

    split = lambda idxs: torch.utils.data.Subset(dataset, idxs)

    train_loader = DataLoader(split(train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(split(val_idx),   batch_size=batch_size, shuffle=False)
    cal_loader   = DataLoader(split(cal_idx),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(split(test_idx),  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, cal_loader, test_loader, X_tensor.shape




# def get_data_loaders(
#         dataset_path,
#         batch_size,
#         *,
#         test_size = 0.2,
#         val_size  = 0.1,
#         cal_size  = 0.2,
#         seed      = 42,
#         y_pos_threshold: float = 0.0,
# ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, tuple]:
#     """
#     Split data within each CASE so every outage case appears in train, val, cal, test.
#     """
#
#     # --------------------------- load ----------------------------
#     X, Y = load_dataset(dataset_path)
#
#     X_tensor = torch.as_tensor(X, dtype=torch.float32)
#     Y_tensor_raw = torch.as_tensor(Y, dtype=torch.float32)
#
#     if X_tensor.dim() == 3:                          # (B, T, N)
#         X_tensor = X_tensor.unsqueeze(2)             # (B, T, 1, N)
#     elif X_tensor.dim() != 4:
#         raise ValueError(f"Unsupported X shape: {X_tensor.shape}")
#
#     # --- Case key BEFORE normalization ---
#     Y_bin = (Y_tensor_raw > y_pos_threshold)
#     case_keys = [tuple(torch.where(Y_bin[i])[0].tolist()) for i in range(Y_bin.shape[0])]
#
#     # Group indices by case
#     groups: Dict[tuple, List[int]] = {}
#     for i, k in enumerate(case_keys):
#         groups.setdefault(k, []).append(i)
#
#     # Normalize labels row-wise to sum=1
#     Y_sum = Y_tensor_raw.sum(dim=1, keepdim=True).clamp(min=1.0)
#     Y_tensor = Y_tensor_raw / Y_sum
#
#     dataset = TensorDataset(X_tensor, Y_tensor)
#
#     if (test_size + val_size + cal_size) >= 1.0:
#         raise ValueError("test_size + val_size + cal_size must be < 1.0")
#
#     g = torch.Generator().manual_seed(seed)
#
#     split_indices = {"train": [], "val": [], "cal": [], "test": []}
#
#     for k, idxs in groups.items():
#         idxs = torch.as_tensor(idxs, dtype=torch.long)
#         perm = idxs[torch.randperm(len(idxs), generator=g)]
#
#         n_total = len(perm)
#         n_test = int(n_total * test_size)
#         n_val  = int(n_total * val_size)
#         n_cal  = int(n_total * cal_size)
#         n_train = n_total - n_test - n_val - n_cal
#
#         split_indices["train"].extend(perm[:n_train].tolist())
#         split_indices["val"].extend(perm[n_train:n_train+n_val].tolist())
#         split_indices["cal"].extend(perm[n_train+n_val:n_train+n_val+n_cal].tolist())
#         split_indices["test"].extend(perm[n_train+n_val+n_cal:].tolist())
#
#     def mk_loader(idxs, shuffle):
#         subset = torch.utils.data.Subset(dataset, idxs)
#         return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
#
#     train_loader = mk_loader(split_indices["train"], shuffle=True)
#     val_loader   = mk_loader(split_indices["val"],   shuffle=False)
#     cal_loader   = mk_loader(split_indices["cal"],   shuffle=False)
#     test_loader  = mk_loader(split_indices["test"],  shuffle=False)
#
#     return train_loader, val_loader, cal_loader, test_loader, X_tensor.shape


# for debugging
def count_case(loader, case_tuple, y_pos_threshold=0.0):
    """Count how many samples in loader belong to a given outage case."""
    count = 0
    for xb, yb in loader:
        # binarize labels
        y_bin = (yb > y_pos_threshold)
        # build case key for each sample in the batch
        keys = [tuple(torch.where(row)[0].tolist()) for row in y_bin]
        # count matches
        count += sum(1 for k in keys if tuple(k) == tuple(case_tuple))
    return count


def pick_case_cp_cover_but_not_topk(P_tst, Y_tst, cp_mask, k=3, y_thr=0.1, prefer_small_set=True):
    """
    Return an index i such that:
      - CP covers all true labels in case i
      - Top-k predictions do NOT cover all true labels in case i
    If none exists, returns None.
    """
    candidates = []
    N = len(P_tst)

    for i in range(N):
        p = np.asarray(P_tst[i], float)
        y = np.asarray(Y_tst[i], float)
        S = set(np.flatnonzero(cp_mask[i]))
        true_idx = set(np.flatnonzero(y > y_thr))
        if not true_idx:       # skip cases with no positives
            continue

        topk = set(np.argsort(p)[::-1][:k])

        cp_covers   = true_idx.issubset(S)
        topk_covers = true_idx.issubset(topk)

        if cp_covers and not topk_covers:
            candidates.append((i, len(S)))

    if not candidates:
        return None

    if prefer_small_set:
        # pick the example with the smallest |S|
        candidates.sort(key=lambda t: t[1])
        return candidates[0][0]
    else:
        return random.choice(candidates)[0]