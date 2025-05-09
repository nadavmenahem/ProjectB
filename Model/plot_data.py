import os
import numpy as np

from data_utils import get_config, load_dataset, get_meta_data
from plot_utils import plot_all_buses


PLOT_CLEAN_DATA = False


if __name__ == "__main__":
    config = get_config()
    
    if PLOT_CLEAN_DATA:
        dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)
    else:
        dataset_path = os.path.join(config.dataset.noisy_path, config.dataset.network_name)

    metadata = get_meta_data(dataset_path)
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