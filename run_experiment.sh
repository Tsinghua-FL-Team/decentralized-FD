#!/bin/bash


cmdargs=$1

# Run Experiment Locally with following configuration
RESULTS_PATH="results/"
DATA_PATH="data/"
CHECKPOINT_PATH="checkpoints/"

python -u code/federated_learning.py --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs
