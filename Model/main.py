#===================IMPORTS==================
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import json
from box import Box
import yaml
import torch
import torch.nn as nn

from plot import plot_graph, plot_data, plot_test_case_probs
from model import SpectralGCN
from data_utils import get_data_loaders, load_dataset
from model_utils import train_model, evaluate_model


#==================CONFIG==================
META_FILE = "meta.json"
GRAPH_FILE = "graph.npy"
CONFIG_FILE = "config.yaml"

PLOTTING = True  # Set to True to plot the data
DEBUGGING = False  # Set to True to enable debugging mode

#==================FUNCTIONS==================

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


#=====================MAIN====================
def main():
    config = get_config()
    dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)
    metadata = get_meta_data(dataset_path)
    G = get_graph(dataset_path)

    train_loader, test_loader, X_shape = get_data_loaders(dataset_path, config.batch_size)
    num_input_features = X_shape[2]  # K

    if DEBUGGING:  
        print("DEBUGGING: X shape:", X_shape)
        print("DEBUGGING: num_input_features:", num_input_features)

    model = SpectralGCN(
        num_nodes=G.number_of_nodes(),
        in_features=num_input_features, # K
        out_features=config.output_features, # G
        G=G, # graph
        H=config.poly_order, # H
        num_classes=G.number_of_edges() # one output per power line
    )

    if DEBUGGING:
        for name, param in model.named_parameters():
            print(f"{name:30} | requires_grad: {param.requires_grad}")


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # need to take from config ~nadav
    criterion = nn.KLDivLoss(reduction="batchmean")

    train_model(model, train_loader, optimizer, criterion, config.num_epochs)

    print("\nEvaluating on test set...")
    y_probs, y_true, y_pred = evaluate_model(model, test_loader)

    print("\nGround truth fault labels for test cases:")

    for i, labels in enumerate(y_true):
        faulty_lines = np.where(labels > 0)[0]  # indices where line is faulty
        print(f"Test case {i}: Faulty lines = {faulty_lines.tolist()}")
        print("prediction: ", y_pred[i])

    if PLOTTING:
        for i in range(len(y_probs)):
            plot_test_case_probs(y_probs[i], y_true[i], i)

    if DEBUGGING:
        # X, Y = load_dataset(dataset_path)
        # for i, y in enumerate(Y):
        #     print(f"Sample {i}: sum = {np.sum(y)}, y = {y}")
        print("DEBUGGING: number of nodes:", G.number_of_nodes())
        print("DEBUGGING: number of edges:", G.number_of_edges())

    if PLOTTING:
        X, Y = load_dataset(dataset_path)
        case_number = 3
        bus_index = np.argmax(Y[case_number])  # Find the index of the bus with a fault
        print(f"Bus index: {bus_index}")

        sampling_rate = metadata["sampling_rate"]
        total_time = metadata["total_time"]
        outage_time = metadata["outage_time"]

        plot_data(X[case_number], bus_index=3, sampling_rate=sampling_rate, total_time=total_time, outage_time=outage_time)

        # for i in range(X.shape[2]):
        #     plot_data(X[case_number], bus_index=i, sampling_rate=sampling_rate, total_time=total_time, outage_time=outage_time)


if __name__ == "__main__":
    main()
