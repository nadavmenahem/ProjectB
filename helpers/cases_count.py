import os
import numpy as np

# Folder containing your .npz files
dataset_dir = 'datasets/N-2_random_outage_time/ieee14/cases'

# Filter for .npz files
files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]

print(f"There are {len(files)} cases in the dataset.")
