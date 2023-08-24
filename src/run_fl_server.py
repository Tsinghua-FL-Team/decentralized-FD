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

    # create a server
    custom_server = server.Server(
        client_manager=client_manager, 
        strategy=agg_strat
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

    print("\n\n")
    all_pre_accuracies = []
    all_pos_accuracies = []
    for item in custom_server.fit_metrics:
        preacc = [0] * len(item)
        posacc = [0] * len(item)
        for client_data in item:
            preacc[client_data["client_id"]] = client_data["pr_accuracy"]
            posacc[client_data["client_id"]] = client_data["ps_accuracy"]
        all_pre_accuracies.append(preacc)
        all_pos_accuracies.append(posacc)
    
    # print results
    for i in range(len(all_pre_accuracies)):
        print(f"D -> {all_pre_accuracies[i]}\n     {all_pos_accuracies[i]}")
    
if __name__ == "__main__":
    main()
