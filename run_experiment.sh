#!/bin/bash


# Activate anaconda environment for flower framework
source /home/${USER}/.bashrc
conda activate fl_flower

# Check for current host name
current_host=$(hostname)

# Run the python based experiment manager
python run_experiment.py --allocated_hosts=$1 --current_host=$current_host --config_file=$2

