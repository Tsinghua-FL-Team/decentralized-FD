#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import argparse, time
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy

#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L O C A L     L I B R A R I E S                          #
#                                                                            #
#----------------------------------------------------------------------------#
from configs.hyperparameters import hyperparams as hp_dicts
import experiment_manager as expm
import models, data
from worker import Worker
from contract import Server


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
args = parser.parse_args()

#DATA_PATH="E:\LAB\experiment\6. Decentralized FD\decentralized-FD\code\datasets\" 
#RESULTS_PATH="E:\LAB\experiment\6. Decentralized FD\decentralized-FD\code\results\" 
#CHECKPOINT_PATH="E:\LAB\experiment\6. Decentralized FD\decentralized-FD\code\checkpoints\"


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   run individual experiment using the information passed.                   #
#                                                                             #
#*****************************************************************************#
def run_experiment(exp, exp_count, n_experiments):
    # print log information
    print("Running Experimemt {} of {} with".format(exp_count+1, n_experiments))
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
    worker_loaders = [DataLoader(local_data, batch_size=hp["batch_size"], shuffle=True) for local_data in worker_data]
    test_loader = DataLoader(test_data, batch_size=hp["batch_size"], shuffle=True)
    distill_loader = DataLoader(distill_data, batch_size=hp["batch_size"])
    
    # create instances of workers and the server (i.e. smart contract)
    workers = [
        Worker(model_fn,
               optimizer_fn, 
               loader,
               idnum = i,
               counts = counts,
               n_classes=hp["n_classes"]) 
        for i, (loader, counts) in enumerate(zip(worker_loaders,label_counts))
        ]
    server = Server(n_samples=len(distill_loader.dataset), 
                    n_classes=hp["n_classes"], 
                    n_workers=hp["n_workers"]
                )
    
    print("Starting Distributed Training..\n")
    t1 = time.time()
    
    # start training each client individually
    for c_round in range(1, hp["communication_rounds"]+1):
        print("Communication Round: " + str(c_round))
        # sample workers for current round of training
        #participating_workers = server.select_workers(workers, hp["participation_rate"])
        #exp.log({"participating_clients" : np.array([worker.id for worker in participating_workers])})
        #participating_workers = workers
        for worker in workers:
            print("Train WORKER: "+str(worker.id))
            
            # get Aggregated Prediction Matrix
            #worker.get_from_server(server)
            
            # local Training / Distillation ??
            train_stats = worker.train(epochs=hp["local_epochs"])
            
            # compute Predictions
            worker.compute_prediction_matrix(loader=distill_loader, 
                                             argmax=True)
            
            # send Predictions + Frequency Vector to Server
            worker.send_to_server(server)
        
        # aggregate the predictions and compute reward
        server.aggregate_and_compute_reward()
        
        # run federated distillation phase
        for worker in workers:
            print("Distill WORKER: "+str(worker.id))
            
            # get Aggregated Prediction Matrix
            worker.get_from_server(server)
            
            # local Training / Distillation ??
            distill_stats = worker.distill(distill_epochs=hp["distill_epochs"],
                                           loader=distill_loader)
                
        # logging the results as described
        if exp.is_log_round(c_round):
            
            print("Experiment: ({}/{})".format(exp_count+1, 
                                               n_experiments)
                  )   
            # log information about communication rounds and epochs
            exp.log({'communication_round' : c_round, 
                     'epochs' : c_round*hp['local_epochs']
                     })
            # log information about parameters on worker
            exp.log({key : workers[0].optimizer.__dict__[
                'param_groups'][0][key] for key in optimizer_hp
                })
            # evaluate and log evaluation results
            for worker in workers:
                # Evaluate each worker's performance 
                exp.log({"worker_{}_{}".format(worker.id, key) : value 
                         for key, value in worker.evaluate(
                                 loader=test_loader).items()
                         })
                  
            # save logging results to disk
            try:
              exp.save_to_disc(path=args.RESULTS_PATH, name=hp['log_path'])
            except:
              print("Saving results Failed!")
    
    # compute total time taken by the experiment
    print("Experiment {} took time {} to run..".format(exp_count, 
                                                       time.time() - t1)
          )
    
    # Free up memory
    del server; workers.clear()
    torch.cuda.empyt_cache()

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   run all experiments as specified by the hyperparameters file.             #
#                                                                             #
#*****************************************************************************#
def run():
    # create instances of experiment manager class for each setup
    experiments = [expm.Experiment(hyperparameters=hp) for hp in hp_dicts]
    
    print("Running a total of {} Experiments..\n".format(len(experiments)))
    for exp_count, experiment in enumerate(experiments):
        run_experiment(experiment, exp_count, len(experiments))

# main program starts here
if __name__ == "__main__":
    run()