#!/bin/bash -l
#SBATCH -A NAISS2023-22-904 -p alvis
#SBATCH -t 0-00:01:00 
#SBATCH --gpus-per-node=T4:4
#SBATCH --nodes=1 --ntasks=1
#SBATCH -J cifar10_modified_run
#SBATCH --output cifar10_modified_run.txt

hostnames=$SLURM_NODELIST
echo "Fetched Hostnames are $hostnames"
wait

apptainer exec --nv /mimer/NOBACKUP/groups/naiss2023-22-904/env_containers/flwr14_env.sif python run_experiment_slurm.py --num_gpus=4 --configs_path="current_experiments/"
