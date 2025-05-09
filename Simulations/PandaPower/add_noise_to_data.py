"""
This script adds *white* noise to the dataset by modifying the 'x' values in the .npz files.
"""


import os
import numpy as np

def add_noise(x):
    signal_std = np.std(x)
    noise_std = signal_std * 0.1  # 10% noise
    noise = np.random.normal(0, noise_std, size=x.shape)
    x_noisy = x + noise
    return x_noisy


def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_cases_dir = os.path.join(input_dir, "cases")
    output_cases_dir = os.path.join(output_dir, "cases")
    os.makedirs(output_cases_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_cases_dir)):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(input_cases_dir, filename))
            x_noisy = add_noise(data['x'])
            y = data['y']

            np.savez(os.path.join(output_cases_dir, filename), x=x_noisy, y=y)
            print(f"Noisy file saved: {filename}")

    # Copy graph.npy and meta.json
    for extra_file in ["graph.npy", "meta.json"]:
        src = os.path.join(input_dir, extra_file)
        dst = os.path.join(output_dir, extra_file)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            print(f"Copied {extra_file}")


if __name__ == "__main__":
    input_dataset = "outage_dataset/ieee14"  # adjust if needed
    output_dataset = "outage_dataset_noisy/ieee14"

    process_dataset(input_dataset, output_dataset)
