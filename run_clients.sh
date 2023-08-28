#!/bin/bash

python src/run_fl_clients.py --total_clients=5 --num_clients=5 --start_cid=0 --max_gpus=4 --config_file="src/configs/exp_configs.yaml"
