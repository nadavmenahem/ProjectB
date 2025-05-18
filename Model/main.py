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
from model_utils import train_model, evaluate_model, save_model, load_model, print_parameter_count
from conformal import ConformalPredictor


#==================CONFIG==================

RESULT_PLOTTING = False  # Set to True to plot the results
DEBUGGING = True  # Set to True to enable debugging mode

#==================FUNCTIONS==================

#=====================MAIN====================
def main():
    parser = argparse.ArgumentParser(description="Spectral GCN for line outage identification")
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to a config.yaml file')
    args = parser.parse_args()
    config = get_config(args.config)

    dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)
    # metadata = get_meta_data(dataset_path)
    G = get_graph(dataset_path)

    train_loader, cal_loader, test_loader, X_shape = get_data_loaders(dataset_path, config.batch_size, config.dataset.test_size, config.dataset.cal_size)
    # X_shape == (B, T, K, N)
    _, time_samples, num_input_features, _ = X_shape

    if DEBUGGING:  
        print("DEBUGGING: X shape:", X_shape)
        print("DEBUGGING: num_input_features:", num_input_features)

    model = SpectralGCN(
        num_nodes=G.number_of_nodes(),
        time_samples=time_samples,           # T
        in_features=num_input_features,      # K
        out_features=config.output_features, # G
        G=G, # graph
        H=config.poly_order, # H
        num_classes=G.number_of_edges() # one output per power line
    )

    if DEBUGGING:
        for name, param in model.named_parameters():
            print(f"{name:30} | requires_grad: {param.requires_grad}")
        print_parameter_count(model)

    # 1. Training the model
    if config.load_pretrained:
        load_model(model=model, params_path=config.params_path, train_loader=train_loader, num_epochs=config.num_epochs)
    else:
        train_model(model=model, train_loader=train_loader, num_epochs=config.num_epochs, params_path=config.params_path)
    
    # 2. Conformal prediction
    conformal = ConformalPredictor(model)

    conformal.calibrate(cal_loader)

    for i, (x_batch, y_batch) in enumerate(test_loader):
        sets, probs, pvals = conformal.predict(x_batch, alpha=config.Conformal.alpha, return_probs=True, return_pvals=True)
        top3_lines = conformal.predict_top_k(x_batch, k=3)

        for j in range(x_batch.shape[0]):
            sample_idx = i * config.batch_size + j
            true_outages = np.where(y_batch[j].cpu().numpy() > 0)[0].tolist()
            predicted_top3 = top3_lines[j].tolist()
            conformal_set = np.where(sets[j])[0].tolist()
            set_size = len(conformal_set)

            print(f"\n🧪 Sample {sample_idx}")
            print(f"  Ground truth: {true_outages}")
            print(f"  Top-3 predicted: {predicted_top3}")
            print(f"  Conformal set: {conformal_set} (size = {set_size})")
            print(f"  Max probability: {np.max(probs[j]):.2f}")



    # 3. Evaluate the model``
    y_probs, y_true, y_pred = evaluate_model(model, test_loader)

    if DEBUGGING:
        print("\nGround truth fault labels for test cases:")
        for i, labels in enumerate(y_true):
            faulty_lines = np.where(labels > 0)[0]  # indices where line is faulty
            print(f"Test case {i}: Faulty lines = {faulty_lines.tolist()}")
            print("prediction: ", y_pred[i])

    if RESULT_PLOTTING:
        for i in range(len(y_probs)):
            plot_test_case_probs(
                y_probs[i],
                y_true[i],
                i,
                topk_indices=top3_lines[i],
                conformal_set=np.where(sets[i])[0].tolist()
            )


    if DEBUGGING:
        # X, Y = load_dataset(dataset_path)
        # for i, y in enumerate(Y):
        #     print(f"Sample {i}: sum = {np.sum(y)}, y = {y}")
        print("DEBUGGING: number of nodes:", G.number_of_nodes())
        print("DEBUGGING: number of edges:", G.number_of_edges())


if __name__ == "__main__":
    main()
