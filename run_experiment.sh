#!/bin/bash


hostnames=$1
configs_file=$2
current_host=$(hostname)


echo $hostnames
echo $current_host



#srun --exclusive --nodelist=s177 --output="server_output" run_server.sh --server_address="0.0.0.0:59999" --config_file="src/configs/exp_configs.yaml" &
#srun --exclusive --output="client_output_m1" run_clients.sh --server_address="s177:59999" --total_clients=5 --num_clients=2 --start_cid=0 --config_file="src/configs/exp_configs.yaml" &
#srun --exclusive --output="client_output_m2" run_clients.sh --server_address="s177:59999" --total_clients=5 --num_clients=2 --start_cid=2 --config_file="src/configs/exp_configs.yaml" &
#srun --exclusive --output="client_output_m3" run_clients.sh --server_address="s177:59999" --total_clients=5 --num_clients=1 --start_cid=4 --config_file="src/configs/exp_configs.yaml" &
