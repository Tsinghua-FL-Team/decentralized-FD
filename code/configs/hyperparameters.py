#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#

#################   Hyperparameters Key   #################

# model                -    string    - Choose from: [simple-cnn, mlp-mnist]

# dataset              -    string    - Choose from: [mnist, cifar10, cifar100]

# distill-dataset      -    string    - Choose from: [stl-10, emnist] 
#                                       if skipped, the test set of dataset
#                                       will be used to extract n_distill
#                                       samples as distillation dataset.

# n_workers            -     int      - Number of Workers.

# worker_data          -   list[int]  - Number of data samples on each worker,
#                                       if not provided then dirichlet function
#                                       will be used, in which case "alpha" is 
#                                       required parameter.

# classes_per_worker    -     int     - Number of different classes every worker 
#                                       holds in it's local data, 0 returns an iid
#                                       split. Works in conjunction with worker_data.

# alpha                 -    float    - alpha parameter for dirichlet distribution
#                                       required if worker_data splits not defined,
#                                       used only if worker_data not defined.

# batch_size            -     int     - Batch-size used by the Workers.

# communication_rounds  -     int     - Total number of communication rounds.

# local_epochs          -     int     - Local training epochs at every worker.

# early_stop            - list[float] - Early stopping accuracy, -1 means no criteria
#                                       must be a list of size n_workers.

# distill_epochs        -     int     - Number of epochs used for distillation.

# n_distill             -     int     - Size of the distilation dataset. Needed
#                                       only if distill-dataset not defined.

# random_seed           -     int     - Random seed for model initializations.

# log_path              -    string   - Path to store the log files.


#################   Unused Hyperparameters   #################

# data_balance          -    float    - Default 1.0, if <1.0 data will be more 
#                                       concentrated on some clients

# participation_rate    -    float    - Fraction of Workers which participate in 
#                                       every Communication Round

# distill_mode          -    string   - The distillation mode, chosse from 
#                                       ("regular", "pate", "pate_up", ..)

# aggregation_mode      -    string   - Aggregation mode used to synchronize
#                                       Choose from: ["FA", "FD"]

# log_frequency         -     int     - Number of communication rounds after which results 
#                                       are logged and saved to disk


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
#          # Experiment 1 [Datasize vs Reward Graph]
#          "model": "lenet_mnist",
#          "dataset": "mnist",
# #         "distill-dataset": "stl10",
#          "n_classes": 10,
#          "n_workers": 9,
#          "classes_per_worker": 0,
#          "worker_data": [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800],
#          "alpha": 0.1,
#          "batch_size": 32,
#          "communication_rounds": 1,
#          "local_epochs": 5,
#          "early_stop": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#          "distill_epochs": 5,
#          "n_distill": 1500,
#          "random_seed": 42,
#          "log_path": "experiment1\\"
#         }, 
        {
         # Experiment 2
         "model": "lenet_mnist",
         "dataset": "mnist",
#         "distill-dataset": "stl10",
         "n_classes": 10,
         "n_workers": 9,
         "classes_per_worker": 0,
         "worker_data": [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800],
         "alpha": 0.1,
         "batch_size": 32,
         "communication_rounds": 1,
         "local_epochs": 5,
         "early_stop": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
         "distill_epochs": 5,
         "n_distill": 1500,
         "random_seed": 42,
         "log_path": "experiment1\\"
        }, 
    ]

