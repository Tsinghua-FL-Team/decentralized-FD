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

# dtrain               -     bool     - whether to use the train portion of
#                                       "distill-dataset" as distillation data.

# n_workers            -     int      - Number of Workers.

# worker_data          -   list[int]  - Number of data samples on each worker,
#                                       if not provided then dirichlet function
#                                       will be used, in which case "alpha" is 
#                                       required parameter.

# classes_per_worker    -     int     - Number of different classes every worker 
#                                       holds in it's local data, 0 returns an iid
#                                       split. Works in conjunction with worker_data.

# alpha                 -    float    - Parameter alpha for dirichlet distribution
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

# r_alpha               -    float    - The scaling factor for reward computation
#                                       mechanism.

# r_beta                -     int     - The penalty term for wrong predictions
#                                       made by a worker compared to its peers.

# use_confidence        -     bool    - Whether to make predictions based on
#                                       confidence measure or not.

# conf_measure          -    float    - Confidence to acheive in order to be 
#                                       able to make predictions.

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
          # Experiment D-3a (Accuracy vs Reward Graph)
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "total_data": 60000,
          "n_distill": 40000,
          "communication_rounds": 10,
          "local_epochs": 10,
          "distill_iter": 3125,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "use_confidence": False,
          "conf_measure": 0.0,
          "random_seed": 42,
          "log_path": "multi_communication/"
        }
    ]

