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
    # Heuristic Branch
    # create a list of possible experimental setups
    global hyperparams
    hyperparams = [
        ###  Heuristic = 10%  / beta = 1  ###
        {
          # Experiment F-1a
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.1,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 20%  / beta = 1  ###
        {
          # Experiment F-1b
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.2,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 30%  / beta = 1  ###
        {
          # Experiment F-1c
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.3,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 40%  / beta = 1  ###
        {
          # Experiment F-1d
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.4,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 50%  / beta = 1  ###
        {
          # Experiment F-1e
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.5,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 60%  / beta = 1  ###
        {
          # Experiment F-1f
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.6,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 70%  / beta = 1  ###
        {
          # Experiment F-1g
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.7,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 80%  / beta = 1  ###
        {
          # Experiment F-1h
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.8,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 90%  / beta = 1  ###
        {
          # Experiment F-1i
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.9,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 100%  / beta = 1  ###
        {
          # Experiment F-1j
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 1,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 1.0,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 

###############################################################################
###############################################################################

        ###  Heuristic = 10%  / beta = 3  ###
        {
          # Experiment F-2a
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.1,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 20%  / beta = 3  ###
        {
          # Experiment F-2b
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.2,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 30%  / beta = 3  ###
        {
          # Experiment F-2c
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.3,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 40%  / beta = 3  ###
        {
          # Experiment F-2d
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.4,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 50%  / beta = 3  ###
        {
          # Experiment F-2e
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.5,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 

        ###  Heuristic = 60%  / beta = 3  ###
        {
          # Experiment F-2f
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.6,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 70%  / beta = 3  ###
        {
          # Experiment F-2g
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.7,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 80%  / beta = 3  ###
        {
          # Experiment F-2h
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.8,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 90%  / beta = 3  ###
        {
          # Experiment F-2i
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.9,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 100%  / beta = 3  ###
        {
          # Experiment F-2j
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 3,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 1.0,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 

###############################################################################
###############################################################################

        ###  Heuristic = 10%  / beta = 5  ###
        {
          # Experiment F-3a
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.1,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 20%  / beta = 5  ###
        {
          # Experiment F-3b
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.2,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 30%  / beta = 5  ###
        {
          # Experiment F-3c
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.3,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 40%  / beta = 5  ###
        {
          # Experiment F-3d
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.4,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 50%  / beta = 5  ###
        {
          # Experiment F-3e
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.5,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 60%  / beta = 5  ###
        {
          # Experiment F-3f
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.6,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 70%  / beta = 5  ###
        {
          # Experiment F-3g
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.7,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 80%  / beta = 5  ###
        {
          # Experiment F-3h
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.8,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 90%  / beta = 5  ###
        {
          # Experiment F-3i
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.9,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 0.30,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
        ###  Heuristic = 100%  / beta = 5  ###
        {
          # Experiment F-3j
          "model": "lenet_mnist",
          "dataset": "emnist",
          "distill-dataset": "mnist",
          "dtrain": True,
          "n_classes": 10,
          "n_workers": 10,
          "classes_per_worker": 0,
          "alpha": 100,
          "r_alpha": 1.0,
          "r_beta": 5,
          "batch_size": 128,
          "communication_rounds": 1,
          "local_epochs": 10,
          "heuristic_%age": 0.5,
          "distill_iter": 500,
          "n_distill": 40000,
          "use_confidence": False,
          "conf_measure": 1.0,
          "random_seed": 42,
          "log_path": "exp_heuristic\\"
        }, 
    ]

