#===================IMPORTS==================
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import argparse

from plot_utils import plot_graph, plot_data, plot_test_case_probs, plot_all_buses
from model import SpectralGCN
from data_utils import get_data_loaders, load_dataset, get_config, get_graph, get_meta_data
from model_utils import train_model, evaluate_model, save_model, load_model


#==================CONFIG==================

DATA_PLOTTING = False  # Set to True to plot the data
RESULT_PLOTTING = True  # Set to True to plot the results
DEBUGGING = False  # Set to True to enable debugging mode

#==================FUNCTIONS==================

#=====================MAIN====================
def main():
    parser = argparse.ArgumentParser(description="Spectral GCN for line outage identification")
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to a config.yaml file')
    args = parser.parse_args()
    config = get_config(args.config)

    dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)
    metadata = get_meta_data(dataset_path)
    G = get_graph(dataset_path)

    train_loader, test_loader, X_shape = get_data_loaders(dataset_path, config.batch_size, config.dataset.test_size)
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

    load_model(model=model, config=config, train_loader=train_loader)

    y_probs, y_true, y_pred = evaluate_model(model, test_loader)

    if DEBUGGING:
        print("\nGround truth fault labels for test cases:")
        for i, labels in enumerate(y_true):
            faulty_lines = np.where(labels > 0)[0]  # indices where line is faulty
            print(f"Test case {i}: Faulty lines = {faulty_lines.tolist()}")
            print("prediction: ", y_pred[i])

    if RESULT_PLOTTING:
        for i in range(len(y_probs)):
            plot_test_case_probs(y_probs[i], y_true[i], i)

    if DEBUGGING:
        # X, Y = load_dataset(dataset_path)
        # for i, y in enumerate(Y):
        #     print(f"Sample {i}: sum = {np.sum(y)}, y = {y}")
        print("DEBUGGING: number of nodes:", G.number_of_nodes())
        print("DEBUGGING: number of edges:", G.number_of_edges())

    if DATA_PLOTTING:
        X, Y = load_dataset(dataset_path)
        case_number = 3
        bus_index = np.argmax(Y[case_number])  # Find the index of the bus with a fault
        print(f"Bus index: {bus_index}")

        sampling_rate = metadata["sampling_rate"]
        total_time = metadata["total_time"]
        outage_time = metadata["outage_time"]

        case_data = X[case_number]  # shape: (T, 1, N) or (T, N)

        if case_data.ndim == 3:
            case_data = case_data.squeeze(1)  # remove feature dimension if needed

        plot_all_buses(X[case_number], total_time=total_time, sampling_rate=sampling_rate, outage_time=outage_time)

        # plot_data(X[case_number], bus_index=3, sampling_rate=sampling_rate, total_time=total_time,
        #            outage_time=outage_time, fixed_ylim=y_lim)


if __name__ == "__main__":
    main()
