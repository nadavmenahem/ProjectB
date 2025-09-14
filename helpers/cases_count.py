import os
import argparse

from Model.utils.data_utils import get_config


parser = argparse.ArgumentParser(description="Spectral GCN for line outage identification")
parser.add_argument('-c', '--config', type=str, default=None,
                    help='Path to a config.yaml file')
args = parser.parse_args()
config = get_config(args.config)

dataset_path = os.path.join(config.dataset.path, config.dataset.network_name, "cases")

# Filter for .npz files
files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]

print(f"There are {len(files)} cases in the dataset.")