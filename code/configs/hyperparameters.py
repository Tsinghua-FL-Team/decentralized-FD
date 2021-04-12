#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#                                                                             #
#   Define hyperparameters that will assit in running experiments.            #
#                                                                             #
#-----------------------------------------------------------------------------#
def init( ):
    # create a list of possible experimental setups
    global hyperparams
    hyperparams = [
        {
         # Experiment 1
         "model": "vgg11",
         "dataset": "cifar10",
         "distill-dataset": "stl10",
         "n_classes": 10,
         "n_workers": 5,
         "classes_per_worker": 4,
         "participation_rate": 1,
         "batch_size": 64,
         "data_balance": 1.0,
         "communication_rounds": 2,
         "local_epochs": 15,
         "distill_epochs": 5,
         "n_distill": 512,
         "distill_mode": "regular",
         "aggregation_mode": "FD",
         "random_seed": 42,
         "log_frequency": 1,
         "log_path": "experiment1\\"
        }, 
        {
         # Experiment 2
         "model": "vgg16",
         "dataset": "cifar10",
         "distill-dataset": "stl10",
         "n_workers": 2,
         "classes_per_worker": 0,
         "participation_rate": 1,
         "batch_size": 32,
         "data_balance": 1.0,
         "communication_rounds": 10,
         "local_epochs": 25,
         "distill_epochs": 10,
         "n_distill": 512,
         "distill_mode": "regular",
         "aggregation_mode": "FD",
         "random_seed": 42,
         "log_frequency": 1,
         "log_path": "experiment2\\"
         }
    ]



#################   Hyperparameters Key   #################

# model                 - Choose from: [simple-cnn, mlp-mnist]
# dataset               - Choose from: [mnist, cifar10, cifar100, emnist]
# distill-dataset       - Choose from: [stl-10]
# n_workers             - Number of Workers
# classes_per_worker    - Number of different Classes every Worker holds 
#                         in it's local data, 0 returns an iid split
# participation_rate    - Fraction of Workers which participate in every 
#                         Communication Round
# batch_size            - Batch-size used by the Workers
# data_balance          - Default 1.0, if <1.0 data will be more 
#                         concentrated on some clients
# communication_rounds  - Total number of communication rounds
# local_epochs          - Local training epochs at every client
# distill_epochs        - Number of epochs used for distillation
# n_distill             - Size of the distilation dataset 
# distill_mode          - The distillation mode, chosse from 
#                         ("regular", "pate", "pate_up", ..)
# aggregation_mode      - Aggregation mode used to synchronize
#                         Choose from: ["FA", "FD"]
# random_seed           - Random seed for model initializations
# log_frequency         - Number of communication rounds after which results 
#                         are logged and saved to disk
# log_path              - path of the log files

