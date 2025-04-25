import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


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


def get_data_loaders(dataset_path, batch_size):
    """
    Load the dataset and create data loaders for training and testing.
    """
    X, Y = load_dataset(dataset_path)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    shape = X_tensor.shape

    if len(shape) == 3:
        # X shape: (B, T, N) → single feature, so add K=1
        X_tensor = X_tensor.unsqueeze(2)  # (B, T, 1, N)

    elif len(shape) != 4:
        raise ValueError(f"Unsupported X shape: {shape}")
    
    # Normalize Y tensor
    Y_sum = Y_tensor.sum(dim=1, keepdim=True)
    Y_sum[Y_sum == 0] = 1  # avoid division by zero
    Y_tensor = Y_tensor / Y_sum
    
    print("\n📦 After normalization:") # ~nadav
    for i in range(len(Y_tensor)):
        print(f"Sample {i}: sum = {Y_tensor[i].sum().item()}")


    # Create a dataset and split into train/test sets
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset)) # need to take from config ~nadav
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffles every epoch
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_tensor.shape