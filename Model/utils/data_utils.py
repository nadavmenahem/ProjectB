import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import networkx as nx
import json
from box import Box
import yaml
from typing import Optional
from pathlib import Path
from typing import Tuple


CONFIG_FILE = "./Model/config.yaml"
GRAPH_FILE = "graph.npy"
META_FILE = "meta.json"


"""
should I turn this into a class?
function that return the shape of the data instead of in the data loader?
"""


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

    case_dir = os.path.join(dataset_path, "cases") # change ~nadav
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


# def get_data_loaders(dataset_path, batch_size, test_size=0.2, cal_size=0.2):
#     """
#     Load the dataset and create data loaders for training and testing.
#     """
#     X, Y = load_dataset(dataset_path)
#
#     # Convert to PyTorch tensors
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     Y_tensor = torch.tensor(Y, dtype=torch.float32)
#
#     shape = X_tensor.shape
#
#     if len(shape) == 3:
#         # X shape: (B, T, N) → single feature, so add K=1
#         X_tensor = X_tensor.unsqueeze(2)  # (B, T, 1, N)
#
#     elif len(shape) != 4:
#         raise ValueError(f"Unsupported X shape: {shape}")
#
#     # Normalize Y tensor
#     Y_sum = Y_tensor.sum(dim=1, keepdim=True)
#     Y_sum[Y_sum == 0] = 1  # avoid division by zero
#     Y_tensor = Y_tensor / Y_sum
#
#     # print("\n📦 After normalization:") # ~nadav
#     # for i in range(len(Y_tensor)):
#     #     print(f"Sample {i}: sum = {Y_tensor[i].sum().item()}")
#
#
#     # Create a dataset and split into train/test sets
#     dataset = TensorDataset(X_tensor, Y_tensor)
#
#     num_samples = len(dataset)
#     torch.manual_seed(42)
#     indices = torch.randperm(num_samples)
#
#     # Split the indices into train and test sets
#     test_size = int(num_samples * test_size)
#     cal_size = int(num_samples * cal_size)
#     train_size = num_samples - test_size - cal_size
#
#     train_indices = indices[:train_size]
#     cal_indices = indices[train_size:train_size + cal_size]
#     test_indices = indices[train_size + cal_size:]
#
#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     cal_dataset = torch.utils.data.Subset(dataset, cal_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)
#
#     # print("\n📦 Test set ground-truth (Y):") # ~nadav
#     # for i in range(len(test_dataset)):
#     #     y = test_dataset[i][1].numpy()
#     #     print(f"Test sample {i}: sum = {np.sum(y):.2f}, y = {np.round(y, 2)}")
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffles every epoch
#     cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, cal_loader, test_loader, X_tensor.shape

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