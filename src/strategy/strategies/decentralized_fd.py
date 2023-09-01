"""Implementation of Federated Average (FedAvg) strategy."""

import numpy as np
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

class DecentralizedFederatedDistillation(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_classes: int,
        num_samples: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        reward_scale_alpha: int = 1.0,
        penalty_term_beta: int = 1.0,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.number_of_classes = num_classes
        self.number_of_samples = num_samples
        self.alpha = reward_scale_alpha
        self.beta = penalty_term_beta
    
    def __repr__(self) -> str:
        return "FederatedAverage"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
        self, server_round: int, 
        majority_votes, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(ndarrays_to_parameters([majority_votes]), config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, None, None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, None, None

        # collect client_ids
        results_by_client_id = [(res.metrics["client_id"], res) for _, res in results]
        sorted_results = sorted(results_by_client_id, key=lambda tup: tup[0])

        def my_argmax(a):
            rows = np.where(a == a.max(axis=1)[:, None])[0]
            rows_multiple_max = rows[:-1][rows[:-1] == rows[1:]]
            my_argmax = a.argmax(axis=1)
            my_argmax[rows_multiple_max] = -1
            return my_argmax

        # Convert results
        predictions = [parameters_to_ndarrays(fit_res.parameters)[0] for (_, fit_res) in sorted_results]
        pred_argmax = [my_argmax(item) for item in predictions]

        class_votes = np.sum(predictions, axis=0)
        class_sums = np.sum(class_votes, axis=0)
        per_worker_sums = [np.sum(worker_prediction, axis=0) for worker_prediction in predictions]
        per_worker_pred = [np.count_nonzero(np.sum(worker_prediction, axis=1) > 0) for worker_prediction in predictions]
        total_predicted = np.sum(per_worker_pred)
        
        # Compute the reward for each client based
        # on their contribution and majority vote
        rewardShare = np.zeros(len(predictions))

        for j in range(len(predictions[0])):
            for i in range(len(predictions)):
                # skip if no prediction made by worker i
                if pred_argmax[i][j] == -1:
                    continue
                # Compute R_i
                Ri = (1.0/(total_predicted-per_worker_pred[i])) * (class_sums - per_worker_sums[i])
                t0 = 0
                nPeers = 0
                # Reward worker i for each peer p
                for p in range(len(predictions)):
                    # Skip if same worker
                    if i == p:
                        continue
                    # skip if no prediction made by peer worker p
                    if pred_argmax[p][j] == -1:
                        continue
                    nPeers += 1
                    # Compute reward
                    t0 += ((1.0/Ri[pred_argmax[i][j]]) - self.beta) if pred_argmax[i][j] == pred_argmax[p][j] else (-1 * self.beta)

                # Reward Share for worker i
                nPeers = 1 if nPeers == 0 else nPeers
                rewardShare[i] += self.alpha * (1.0/nPeers) * t0

        majority_vote = np.argmax(class_votes, axis=-1).astype("uint8")
        
        # Collect metrics from clients and store them to disk
        organized_metrics = dict()
        fit_metrics = [res.metrics for (_, res) in sorted_results]

        for client_dict in fit_metrics:
            organized_metrics[f"client_{client_dict['client_id']}_disti_acc"] = client_dict["distill_accuracy"]
            organized_metrics[f"client_{client_dict['client_id']}_codis_acc"] = client_dict["codistill_accuracy"]
            organized_metrics[f"client_{client_dict['client_id']}_train_acc"] = client_dict["train_accuracy"]
            organized_metrics[f"client_{client_dict['client_id']}_disti_loss"] = client_dict["distill_loss"]
            organized_metrics[f"client_{client_dict['client_id']}_codis_loss"] = client_dict["codistill_loss"]
            organized_metrics[f"client_{client_dict['client_id']}_train_loss"] = client_dict["train_loss"]
            organized_metrics[f"client_{client_dict['client_id']}_reward"] = rewardShare[client_dict['client_id']]

        return majority_vote, rewardShare, organized_metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            # Let's assume we won't perform the global model evaluation on the server side.
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
