#===================IMPORTS==================
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn

from GFT import gft
from plot import plot_graph, plot_data


#==================CONSTANS==================
DATA_ROOT = "outage_dataset"
NETWORK_NAME = "ieee14"  # Change this to the desired network name
GRAPH_FILE = "graph.npy"
META_FILE = "meta.json"

PLOTTING = True  # Set to True to plot the data


#==================FUNCTIONS==================
#==================DATASET====================
def load_dataset():
    """
    Load all case simulations for the selected network.
    """
    print("Loading dataset...")

    case_dir = os.path.join(DATA_ROOT, NETWORK_NAME, "cases")
    X, Y = [], []

    for filename in sorted(os.listdir(case_dir)):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(case_dir, filename))
            X.append(data["x"])
            Y.append(data["y"])

    X = np.stack(X)  # shape: [num_cases, timesteps, num_buses]
    Y = np.stack(Y)  # shape: [num_cases, num_lines]

    return X, Y


def get_graph():
    """
    Load the graph topology from the graph.npy file.
    """
    path = os.path.join(DATA_ROOT, NETWORK_NAME, GRAPH_FILE)
    edge_index = np.load(path)

    edge_list = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_edges_from(edge_list)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def get_meta_data():
    """
    Load the metadata from the meta.json file.
    """
    path = os.path.join(DATA_ROOT, NETWORK_NAME, META_FILE)
    with open(path, "r") as f:
        metadata = json.load(f)

    return metadata


#====================MODEL====================
class SpectralConvLayer(nn.Module):
    def __init__(self, in_features, out_features, Lambda, H=1):
        super().__init__()
        self.H = H
        self.in_features = in_features
        self.out_features = out_features

        # Lambda is the eigenvalue matrix (diagonal), shape: [N]
        self.register_buffer("Lambda", Lambda)

        # Learnable parameters: one set of weights per hop degree h
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, out_features)) for _ in range(H + 1)
        ])


    def forward(self, X_hat):
        # X_hat shape: [batch, N, in_features]
        batch_size, N, _ = X_hat.shape
        Y_hat = 0

        for h in range(self.H + 1):
            Λh = self.Lambda ** h  # shape: [N]
            Λh_diag = torch.diag(Λh)  # [N, N]

            # Apply Λ^h to signal
            filtered = torch.matmul(Λh_diag, X_hat)  # [batch, N, in_features]
            Y_hat += torch.matmul(filtered, self.weights[h])  # [batch, N, out_features]

        return Y_hat


class SpectralGCN(nn.Module):
    def __init__(self, V, Lambda, in_features, hidden_features, out_features, H=1):
        super().__init__()
        self.V = V           # Eigenvectors, shape [N, N]
        self.VT = V.T        # GFT matrix
        self.Lambda = Lambda # Eigenvalues, shape [N]

        self.conv1 = SpectralConvLayer(in_features, hidden_features, Lambda, H)
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # x shape: [batch, time, N]
        batch_size, T, N = x.shape

        # Transpose to [batch, N, time]
        x = x.transpose(1, 2)

        # Apply GFT to each node’s signal
        x_hat = torch.matmul(self.VT, x)  # shape: [batch, N, time]

        # GCN expects last dim to be features → [batch, N, time]
        out_hat = self.conv1(x_hat)

        # Aggregate (e.g., mean over nodes or time), or flatten
        out = out_hat.mean(dim=1)  # [batch, hidden_features]
        return torch.sigmoid(self.fc(out))  # Binary classifier per line


#=====================MAIN====================
def main():
    metadata = get_meta_data()

    G = get_graph()
    print("Graph loaded.")

    X, Y = load_dataset()
    print("Dataset loaded.")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    gft_sig = gft(G, X[0][0])
    # print("GFT sig: ", gft_sig)

    if PLOTTING:
        case_number = 3
        bus_index = np.argmax(Y[case_number])  # Find the index of the bus with a fault
        print(f"Bus index: {bus_index}")

        plot_graph(G, numbering=True, faulty_lines=Y[case_number])
        # plot_graph(G, signal=X[0][0])
        # plot_graph(G, signal=gft_sig)

        sampling_rate = metadata["sampling_rate"]
        total_time = metadata["total_time"]
        # plot_data(X[case_number], bus_index=4, sampling_rate=sampling_rate, total_time=total_time)
        
        for i in range(X[0][0].shape[0]):
            plot_data(X[case_number], bus_index=i, sampling_rate=sampling_rate, total_time=total_time)


if __name__ == "__main__":
    main()
