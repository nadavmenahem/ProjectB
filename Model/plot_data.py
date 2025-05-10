import os
import numpy as np
import argparse

from data_utils import get_config, load_dataset, get_meta_data
from plot_utils import plot_all_buses

CASE_NUMBER = 60  # The case number to plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral GCN for line outage identification")
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to a config.yaml file')
    args = parser.parse_args()
    config = get_config(args.config)
    
    dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)

    metadata = get_meta_data(dataset_path)
    X, Y = load_dataset(dataset_path)
    line_index = np.argmax(Y[CASE_NUMBER])  # Find the index of the bus with a fault
    print(f"line outage at line: {line_index}")

    sampling_rate = metadata["sampling_rate"]
    total_time = metadata["total_time"]
    outage_time = metadata["outage_time"]

    case_data = X[CASE_NUMBER]  # shape: (T, 1, N) or (T, N)

    if case_data.ndim == 3:
        case_data = case_data.squeeze(1)  # remove feature dimension if needed

    plot_all_buses(X[CASE_NUMBER], total_time=total_time, sampling_rate=sampling_rate, outage_time=outage_time)