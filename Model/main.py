#===================IMPORTS==================
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import json
from box import Box
import yaml

from graph_utils import graph_spectral_decomposition
from plot import plot_graph, plot_data
from model import SpectralGCN

#==================CONFIG==================
META_FILE = "meta.json"
GRAPH_FILE = "graph.npy"
CONFIG_FILE = "config.yaml"

PLOTTING = False  # Set to True to plot the data
DEBUGGING = True  # Set to True to enable debugging mode

#==================FUNCTIONS==================
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

    return X, Y


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


# Fix ~nadav
def get_config():
    """
    Load the configuration from the config.yaml file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, CONFIG_FILE)
    
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    # print(config.output_features)  # 16

    return config


def get_model(Lambda, num_classes, num_buses):
    config = get_config()

    model = SpectralGCN(
        num_nodes=num_buses,
        in_features=config.input_features,  # K
        out_features=config.output_features,  # G
        Lambda=Lambda,
        H=config.poly_order,  # H
        num_classes=num_classes  # one output per power line
    )

    return model

#=====================MAIN====================
def main():
    config = get_config()
    dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)
    metadata = get_meta_data(dataset_path)
    G = get_graph(dataset_path)
    # num_of_edges = Y.shape[1], num_of_buses = X.shape[2]

    X, Y = load_dataset(dataset_path)
    print("Dataset loaded.")
    
    Lambda, GFTT  = graph_spectral_decomposition(G)

    GFT = GFTT.T
    gft_sig = GFT @ X[0][0]

    model = get_model(Lambda=Lambda, num_classes=G.number_of_edges(), num_buses=G.number_of_nodes())

    if DEBUGGING:
        print("DEBUGGING: Lambda shape:", Lambda.shape)
        print("DEBUGGING: GFT shape:", GFT.shape)
        print("DEBUGGING: X shape:", X.shape)
        print("DEBUGGING: X[0][0] shape:", X[0][0].shape)
        print("DEBUGGING: X shape[2]:", X.shape[2])
        print("DEBUGGING: Y shape[1]:", Y.shape[1])
        print("DEBUGGING: number of nodes:", G.number_of_nodes())
        print("DEBUGGING: number of edges:", G.number_of_edges())
        print("DEBUGGING: Y shape:", Y.shape)
        # print("DEBUGGING: GFT sig: ", gft_sig)

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
        
        for i in range(X.shape[2]):
            plot_data(X[case_number], bus_index=i, sampling_rate=sampling_rate, total_time=total_time)


if __name__ == "__main__":
    main()
