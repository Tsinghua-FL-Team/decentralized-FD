"""A function to load the desired aggregation strategy."""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

import flwr as fl

import sys
import os
sys.path.append(os.path.abspath("src"))
import datasets
import models
import modules


def get_strategy(user_configs: dict):
    # Check what device to use on 
    # server side to run the computations
    run_device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if user_configs["SERVER_CONFIGS"]["RUN_DEVICE"] == "auto" \
            else user_configs["SERVER_CONFIGS"]["RUN_DEVICE"]
    
    # Check wether to evaluate the global
    # model on the server side or not
    eval_fn = None

    # Build the fit config function
    fit_config_fn = None
    # fit_config_fn = get_fit_config_fn(
    #     local_epochs=user_configs["CLIENT_CONFIGS"]["LOCAL_EPCH"],
    #     local_batchsize=user_configs["CLIENT_CONFIGS"]["BATCH_SIZE"],
    #     learning_rate=user_configs["CLIENT_CONFIGS"]["LEARN_RATE"],
    # )
    
    # Create an instance of the  desired aggregation strategy
    if user_configs["SERVER_CONFIGS"]["AGGREGATE_STRAT"] == "DECFD":
        from .strategies.decentralized_fd import DecentralizedFederatedDistillation
        stratgy = DecentralizedFederatedDistillation(
            fraction_fit=user_configs["SERVER_CONFIGS"]["SAMPLE_FRACTION"],
            min_fit_clients=user_configs["SERVER_CONFIGS"]["MIN_SAMPLE_SIZE"],
            min_available_clients=user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"],
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
            num_classes=user_configs["MODEL_CONFIGS"]["NUM_CLASSES"],
            num_samples=user_configs["DATASET_CONFIGS"]["DISTILL_SAMPLES"],
            reward_scale_alpha=user_configs["SERVER_CONFIGS"]["REWARD_SCALER_ALPHA"],
            penalty_term_beta=user_configs["SERVER_CONFIGS"]["REWARD_PENALTY_BETA"],
        )
        return stratgy
    else:
        raise Exception(f"Invalid aggregation strategy {user_configs['SERVER_CONFIGS']['AGGREGATE_STRAT']} requested.")

# def get_fit_config_fn(
#         local_epochs, 
#         local_batchsize, 
#         learning_rate, 
#     ):
#     def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
#         """Return a configuration with static batch size and (local) epochs."""
#         config: Dict[str, fl.common.Scalar] = {
#             "epoch_global": str(server_round),
#             "epochs": str(local_epochs),
#             "batch_size": str(local_batchsize),
#             "learning_rate": str(learning_rate),
#         }
#         return config
#     return fit_config
