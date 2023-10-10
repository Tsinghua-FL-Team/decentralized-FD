#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_fraunhofer_email_address_here>
#SBATCH --job-name=first_job
#SBATCH --output=first_job.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --gpus=4
#SBATCH --output run_results_stdout.txt
##----- #SBATCH --time=2:00

echo "hello slurm"
sleep 60
# Put the Container Run command here!
echo "job finished"
