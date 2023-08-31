"""Module to run the Federated Learning server specified by experiment configurations."""

import argparse
from typing import List, Tuple, Union

import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

from modules import server
import configs
import strategy

def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help="gRPC server address (default: [::]:8080)",
    )
    parser.add_argument(
        "--config_file",
        type = str,
        required = True,
        help="Configuration file to use (no default)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()
    user_configs = configs.parse_configs(args.config_file)

    # Create strategy
    agg_strat = strategy.get_strategy(user_configs)

    # create a client manager
    client_manager = SimpleClientManager()

    # display stats function
    display_stats = get_display_stat_function(num_clients=user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"])

    # create a server
    custom_server = server.Server(
        client_manager=client_manager, 
        strategy=agg_strat,
        display_results=display_stats,
    )

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"]),
        server=custom_server,
    )

    # Fetch stats and store them locally?
    import ntpath
    exp_config = ntpath.basename(args.config_file)
    print(exp_config[:-5])

    # Display final stats
    print("\n\n")
    print("Final Results!!!")
    display_stats(custom_server.fit_metrics)
    print("\n\n")

def get_display_stat_function(num_clients):
    def display_stats(fit_metrics):
        all_tr_accu = []
        all_ds_accu = []
        all_co_accu = []
        for item in fit_metrics:
            tr_accu = [0] * num_clients
            ds_accu = [0] * num_clients
            co_accu = [0] * num_clients
            for client_data in item:
                tr_accu[client_data["client_id"]] = client_data["train_accuracy"]
                ds_accu[client_data["client_id"]] = client_data["distill_accuracy"]
                co_accu[client_data["client_id"]] = client_data["codistill_accuracy"]
            # append results
            all_tr_accu.append(tr_accu)
            all_ds_accu.append(ds_accu)
            all_co_accu.append(co_accu)
        
        # print results
        for i in range(len(all_tr_accu)):
            non_zero = sum(x != 0 for x in all_tr_accu[i])
            if non_zero == 0: non_zero += 1
            print(f"T -> {sum(all_tr_accu[i]) / non_zero:.4f} -> {all_tr_accu[i]}")
        
        for i in range(len(all_ds_accu)):
            non_zero = sum(x != 0 for x in all_ds_accu[i])
            if non_zero == 0: non_zero += 1
            print(f"D -> {sum(all_ds_accu[i]) / non_zero:.4f} -> {all_ds_accu[i]}")
        
        for i in range(len(all_co_accu)):
            non_zero = sum(x != 0 for x in all_co_accu[i])
            if non_zero == 0: non_zero += 1
            print(f"C -> {sum(all_co_accu[i]) / non_zero:.4f} -> {all_co_accu[i]}")
    return display_stats

if __name__ == "__main__":
    main()
