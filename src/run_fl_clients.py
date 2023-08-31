from multiprocessing import Process

import time
import argparse

import torch
import flwr as fl

from modules import client
import models
import configs
import datasets

import numpy as np
import random


def client_runner(
        client_id: int,
        total_clients: int,
        config_file: str,
        server_address: str,
        max_gpus: int,
        log_host: str,
    ):
    print(f"Running client {client_id} of {total_clients}.")
    # Configure logger
    fl.common.logger.configure(f"client_{client_id}", host=log_host)

    # Load user configurations
    user_configs = configs.parse_configs(config_file)
    
    # Check for runnable device
    local_device = user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"]
    if local_device == "auto":
        local_device = f"cuda:{int(client_id%max_gpus)}" if torch.cuda.is_available() else "cpu"
    
    random_seeder = get_random_seeder(seed_value=user_configs["MISC_CONFIGS"]["RANDOM_SEED"], use_cuda=True if local_device[:4]=="cuda" else False)
    
    # Load model and data
    model_fn = models.load_model(model_configs=user_configs["MODEL_CONFIGS"])

    random_seeder()
    trainset, distillset, testset = datasets.load_and_fetch_split(
        client_id=client_id,
        n_clients=total_clients,
        dataset_conf=user_configs["DATASET_CONFIGS"],
    )

    # determine early stopping criteria
    es_criteria = user_configs["MISC_CONFIGS"]["EARLY_STOPS"][client_id] if len(user_configs["MISC_CONFIGS"]["EARLY_STOPS"]) > client_id else -1
    
    # determine client type (normal, heuristic, colluding)
    # based on the percentage of clients and current cliend id
    client_type = "normal"
    if (user_configs["MISC_CONFIGS"]["HEURISTIC_RATIO"] > 0 or 
        user_configs["MISC_CONFIGS"]["COLLUSION_RATIO"] > 0):
        n_heuristic = int(user_configs["MISC_CONFIGS"]["HEURISTIC_RATIO"] * total_clients)
        n_colluding = int(user_configs["MISC_CONFIGS"]["COLLUSION_RATIO"] * total_clients)
        if client_id < n_heuristic:
            client_type = "heuristic"
        if client_id < n_colluding:
            client_type = "colluding"

    # Start client
    custom_client = client.Client(
        client_id=client_id,
        model_fn=model_fn,
        trainset=trainset[0],
        testset=testset,
        distillset=distillset,
        early_stop=es_criteria,
        predict_confid=user_configs["CLIENT_CONFIGS"]["PRED_CONF"],
        criterion_str=user_configs["CLIENT_CONFIGS"]["CRITERION"],
        optimizer_str=user_configs["CLIENT_CONFIGS"]["OPTIMIZER"],
        num_classes=user_configs["MODEL_CONFIGS"]["NUM_CLASSES"],
        co_distill_epochs=user_configs["CLIENT_CONFIGS"]["CO_DIST_EPOCH"],
        distill_epochs=user_configs["CLIENT_CONFIGS"]["DIST_EPOCH"],
        trainer_epochs=user_configs["CLIENT_CONFIGS"]["LOCAL_EPCH"],
        learning_rate=user_configs["CLIENT_CONFIGS"]["LEARN_RATE"],
        batch_size=user_configs["CLIENT_CONFIGS"]["BATCH_SIZE"],
        random_seed=user_configs["MISC_CONFIGS"]["RANDOM_SEED"],
        onehot_output=user_configs["CLIENT_CONFIGS"]["ONEHOT_OUT"],
        client_type=client_type,
        rand_seeder=random_seeder,
        device=local_device,
    )
    
    finished = False
    while not finished:
        # try:
        fl.client.start_client(server_address=server_address, client=custom_client)
        finished = True
        # except:
        #     print("Connection Failure! Retrying after 30 seconds.")
        #     finished = True

def get_random_seeder(seed_value, use_cuda):
    def random_seeder(rand_seed=seed_value, cuda=use_cuda):
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        if cuda:
            torch.cuda.manual_seed_all(rand_seed)
    return random_seeder

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="gRPC server address (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="# of clients to run (default: 1)"
    )
    parser.add_argument(
        "--total_clients",
        type=int,
        required=True,
        help="Total number of clients in federation (no default)"
    )
    parser.add_argument(
        "--start_cid",
        type=int,
        required=True,
        help="Client ID for first client (no default)"
    )
    parser.add_argument(
        "--config_file",
        type = str,
        required = True,
        help="Configuration file to use (no default)",
    )
    parser.add_argument(
        "--max_gpus",
        type=int,
        default=1,
        help="Maximum number of available GPUs (default: 1)",
    )
    parser.add_argument(
        "--log_host", type=str, help="Logserver address (no default)",
    )
    args = parser.parse_args()
    client_queue = []
    for cid in range(args.num_clients):
        client_queue.append(Process(target=client_runner, args=(
            cid+args.start_cid,
            args.total_clients,
            args.config_file,
            args.server_address,
            args.max_gpus,
            args.log_host,
        )))
        client_queue[-1].start()
    
    for client_proc in client_queue:
        client_proc.join()

if __name__ == "__main__":
    main()
