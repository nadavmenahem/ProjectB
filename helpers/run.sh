#!/usr/bin/env bash


# to run main:
python Model/main.py --config Model/config.yaml

# to run plot_data:
python Model/plot_data.py --config Model/config.yaml

# to run generate_noisy_data:
python Simulations/PandaPower/generate_noisy_data.py --config Simulations/PandaPower/data_config.yaml


# weird method to add noise
python Simulations/PandaPower/add_noise_to_data2.py datasets/outage_dataset/ieee14 datasets/outage_dataset_noisy_ou/ieee14 --model ou --sigma 0.3 --tau 4.0 --reps 5


# for viewing optuna plots
optuna-dashboard sqlite:///optuna.db --port 8080
# at url:
# http://localhost:8080