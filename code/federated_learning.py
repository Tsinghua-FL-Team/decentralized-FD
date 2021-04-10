#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import os, argparse
import torch
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
import numpy as np

#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
from configs.hyperparameters import hyperparams as hp_dicts
import experiment_manager as expm
import models, data
from worker import Worker


np.set_printoptions(precision=4, suppress=True)

#-----------------------------------------------------------------------------#
#                                                                             #
#   Parse passed arguments to get meta parameters.                            #
#                                                                             #
#-----------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)
#parser.add_argument("--schedule", default="main", type=str)
#parser.add_argument("--start", default=0, type=int)
#parser.add_argument("--end", default=None, type=int)
args = parser.parse_args()


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   run individual experiment using the information passed.                   #
#                                                                             #
#*****************************************************************************#
def run_experiment(exp, exp_count, n_experiments):
    # print log information
    print(exp)
    
    # get hyperparameters of current experiment
    hp = exp.hyperparameters
    model_fn, optimizer, optimizer_hp = models.get_model(hp["model"])
    optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()})
    
    # get datasets needed for training and distillation
    train_data, test_data = data.load_data(hp["dataset"], args.DATA_PATH)
    distill_data = data.load_data(hp["distill-dataset"], args.DATA_PATH)
    
    # setup up random seed as defined in hyperparameters
    np.random.seed(hp["random_seed"])
    
    # distributed the training dataset among worker nodes
    worker_data, label_counts = data.split_data(
        train_data,
        n_workers=hp["n_workers"],
        classes_per_worker=hp["classes_per_worker"]
        )
    
    # create dataloaders for all datasets loaded so far
    worker_loaders = [DataLoader(local_data) for local_data in worker_data]
    test_loader = DataLoader(test_data)
    distill_loader = DataLoader(distill_data)
    
    # create instances of workers and the server (i.e. smart contract)
    workers = [
        Worker(model_fn, 
               optimizer_fn, 
               loader,
               idnum = i,
               counts = counts,
               distill_loader = distill_loader) 
        for i, (loader, counts) in enumerate(zip(worker_loaders,label_counts))
        ]
    
    
    
    return 0


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   run all experiments as specified by the hyperparameters file.             #
#                                                                             #
#*****************************************************************************#
def run():
    # create instances of experiment manager class for each setup
    experiments = [expm.Experiment(hyperparameters=hp) for hp in hp_dicts]
    
    print("Running {} Experiments..\n".format(len(experiments)))
    for exp_count, experiment in enumerate(experiments):
        run_experiment(experiment, exp_count, len(experiments))

# main program starts here
if __name__ == "__main__":
    run()