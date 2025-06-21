#===================IMPORTS==================
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import argparse

from model import SpectralGCN
from model_dft import DFTSpectralGCN
from utils.plot_utils import plot_graph, plot_data, plot_test_case_probs, plot_all_buses
from utils.data_utils import get_data_loaders, load_dataset, get_config, get_graph, get_meta_data
from utils.model_utils import train_model, evaluate_model, save_model, load_model, print_parameter_count
from conformal import ConformalPredictor


# ==================DEVICE==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # remove if you want to use GPU


#==================CONFIG==================
RESULT_PLOTTING = True  # Set to True to plot the results
DEBUGGING = False  # Set to True to enable debugging mode


#=====================MAIN====================
def main():
    print(f"\nUsing device: {device}\n")

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

    params_path = os.path.join(config.model.params_folder, f"{config.model.type}.pth")

    if config.model.type == 'gft':
        model = SpectralGCN(
            num_nodes=G.number_of_nodes(),
            time_samples=time_samples,           # T
            in_features=num_input_features,      # K
            out_features=config.output_features, # G
            G=G, # graph
            H=config.poly_order, # H
            num_classes=G.number_of_edges() # one output per power line
        )

    elif config.model.type == 'dft':
        model = DFTSpectralGCN(
            num_nodes=G.number_of_nodes(),
            time_samples=time_samples,           # T
            in_features=num_input_features,      # K
            out_features=config.output_features, # G
            G=G, # graph
            H=config.poly_order, # H
            num_classes=G.number_of_edges() # one output per power line
        )

    else:
        raise ValueError(f"Unknown model type: {config.model.type}. Supported types are 'gft' and 'dft'.")

    model.to(device)

    # if DEBUGGING:
    if True:
        for name, param in model.named_parameters():
            print(f"{name:30} | requires_grad: {param.requires_grad}")
        print_parameter_count(model)

    # Training the model
    if config.model.load_pretrained:
        load_model(model=model, params_path=params_path)
    else:
        train_model(model, train_loader, config, params_path, test_loader=test_loader)
    
    # Conformal prediction
    conformal = ConformalPredictor(model, loss_name=config.loss_function)

    conformal.calibrate(cal_loader)

    all_sets = []
    all_top3 = []

    for i, (x_batch, y_batch) in enumerate(test_loader):
        sets, probs, pvals = conformal.predict(x_batch, alpha=config.alpha, return_probs=True, return_pvals=True)
        top3_lines = conformal.predict_top_k(x_batch, k=3)

        all_sets.append(sets)
        all_top3.append(top3_lines)

        for j in range(x_batch.shape[0]):
            sample_idx = i * config.batch_size + j
            true_outages = np.where(y_batch[j].cpu().numpy() > 0)[0].tolist()
            predicted_top3 = top3_lines[j].tolist()
            conformal_set = np.where(sets[j])[0].tolist()
            set_size = len(conformal_set)

            if DEBUGGING:
                print(f"\n Sample {sample_idx}")
                print(f"  Ground truth: {true_outages}")
                print(f"  Top-3 predicted: {predicted_top3}")
                print(f"  Conformal set: {conformal_set} (size = {set_size})")
                print(f"  Max probability: {np.max(probs[j]):.2f}")

    # Convert accumulated results to full arrays
    all_sets = np.concatenate(all_sets, axis=0)     # shape: (total_samples, n_lines)
    all_top3 = np.concatenate(all_top3, axis=0)     # shape: (total_samples, 3)

    # Evaluate the model
    y_probs, y_true, topk_idx, correct, total = evaluate_model(
        model, test_loader, config.loss_function, k=config.topk
    )
    if config.model.load_pretrained:
        print(f"Test Top-{config.topk} Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

    if RESULT_PLOTTING:
        for i, (probs, true_labels, topk) in enumerate(zip(y_probs, y_true, topk_idx)):
            plot_test_case_probs(
                probs=probs,                      # model’s predicted probabilities
                true_labels=true_labels,          # ground-truth binary vector
                case_idx=i,                       # index of this test case
                topk_indices=topk.tolist(),       # your top-k predictions
                conformal_set=np.where(all_sets[i])[0].tolist()
            )


    if DEBUGGING:
        # X, Y = load_dataset(dataset_path)
        # for i, y in enumerate(Y):
        #     print(f"Sample {i}: sum = {np.sum(y)}, y = {y}")
        print("DEBUGGING: number of nodes:", G.number_of_nodes())
        print("DEBUGGING: number of edges:", G.number_of_edges())


if __name__ == "__main__":
    main()
