#!/usr/bin/env bash


# to run main:
python Model/main.py --config Model/config.yaml

# to run plot_data:
python Model/plot_data.py --config Model/config.yaml

# to run generate_noisy_data:
python Simulations/PandaPower/generate_noisy_data.py --config Simulations/PandaPower/data_config.yaml