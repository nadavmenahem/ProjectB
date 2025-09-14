import os
import numpy as np
import argparse

from utils.data_utils import get_config, load_dataset, get_meta_data, get_graph
from utils.plot_utils import plot_all_buses, plot_graph

CASE_NUMBER = 3651  # case number to plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral GCN for line outage identification")
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to a config.yaml file')
    args = parser.parse_args()
    config = get_config(args.config)
    
    dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)

    metadata = get_meta_data(dataset_path)
    X, Y = load_dataset(dataset_path)

    probs = Y[CASE_NUMBER]
    line_index = np.argmax(probs)  # Find the index of the outaged line

    outaged_idx = np.where(probs > 1e-6)[0]

    if outaged_idx.size == 0:
        print("no outage detected")
    else:
        print(f"outage at line(s): {outaged_idx.tolist()}")

    sampling_rate = metadata["sampling_rate"]
    total_time = metadata["total_time"]
    outage_time = metadata["outage_time"]

    case_data = X[CASE_NUMBER]  # shape: (T, 1, N) or (T, N)
    T = 20  # Arbitrary time to plot

    if case_data.ndim == 3:
        case_data = case_data.squeeze(1)  # remove feature dimension if needed

    # Plot the graph of the network
    G = get_graph(dataset_path)
    outaged_idx = np.where(np.asarray(probs) > 0)[0]
    # plot_graph(G, signal=case_data[T, :], numbering=True, faulty_lines=outaged_idx)
    plot_graph(G, numbering=True, faulty_lines=outaged_idx)
    plot_all_buses(X[CASE_NUMBER], total_time=total_time, sampling_rate=sampling_rate, outage_time=outage_time)