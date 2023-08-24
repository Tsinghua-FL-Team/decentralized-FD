"""Implementation of Client using Flower Federated Learning Framework"""

import timeit
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import modules

class Client(Client):
    """Represents an honest client.
    Attributes:

    """
    def __init__(
            self, 
            client_id: str,
            local_model: torch.nn.Module,
            trainset: Dataset,
            testset: Dataset,
            distillset: Dataset,
            early_stop: float,
            predict_confid: float,
            criterion_str: str,
            optimizer_str: str,
            num_classes: int,
            distill_epochs: int,
            trainer_epochs: int,
            learning_rate: float,
            batch_size: int,
            client_type: str,
            random_seed: int,
            onehot_output: bool,
            device: str,
            ) -> None:
        """Initializes a new honest client."""
        super().__init__()
        self._client_id = client_id
        self._local_model = local_model
        self._trainset = trainset
        self._testset = testset
        self._distillset = distillset
        self._early_stop = early_stop
        self._predict_confid = predict_confid
        self._optimizer_str = optimizer_str
        self._criterion_str = criterion_str
        self._num_classes = num_classes
        self._distill_epochs = distill_epochs
        self._trainer_epochs = trainer_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._client_type = client_type
        self._random_seed = random_seed
        self._onehot_output = onehot_output
        self._device = device

    @property
    def client_id(self):
        """Returns current client's id."""
        return self._client_id
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Module to fetch model parameters of current client."""
        print(f"[Client {self.client_id}] get_parameters, config: {ins.config}")

        weights = self._local_model.get_weights()
        parameters = ndarrays_to_parameters(weights)
        
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )
    def distill(self, global_labels):
        self._distillset.setTargets(labels=global_labels)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.client_id}] fit, config: {ins.config}")
        fit_begin = timeit.default_timer()
        
        # config = ins.config
        # Get training config
        #local_epochs = int(config["epochs"])
        #batch_size = int(config["batch_size"])
        #learning_rate = float(config["learning_rate"])

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=self._batch_size, shuffle=True
        )
        
        testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=self._batch_size, shuffle=False
        )        
        
        distillloader = torch.utils.data.DataLoader(
            self._distillset, batch_size=self._batch_size, shuffle=False
        )        
            
        criterion = modules.get_criterion(
            criterion_str=self._criterion_str
        )
        optimizer = modules.get_optimizer(
            optimizer_str=self._optimizer_str,            
            local_model=self._local_model,
            learning_rate=self._learning_rate,
        )
        pretrain_loss, pretrain_accuracy = 0.0, 0.0
        posttrain_loss, posttrain_accuracy = 0.0, 0.0

        if self._client_type == "normal":
            ## Normal Client Case - Distill -> Train -> Predict
            public_labels = parameters_to_ndarrays(ins.parameters)
            if len(public_labels[0]) > 0:
                distillloader.dataset.setTargets(labels=public_labels[0][0])
                distill_stats = modules.distill_train(
                    model=self._local_model,
                    distill_loader=distillloader,
                    optimizer=optimizer,
                    device=self._device,
                    num_classes=self._num_classes,
                    distill_epochs=self._distill_epochs,
                )
                ### Predictions Accuracy of the Global Labels
                if self._client_id == 0:
                    print(f"Global Accuracy of the distillation set: {np.count_nonzero(distillloader.dataset.targets == distillloader.dataset.oTargets) / len(distillloader.dataset.oTargets)}")
            
            pretrain_loss, pretrain_accuracy = modules.evaluate(
                model=self._local_model, 
                testloader=testloader, 
                criterion=criterion,
                device=self._device
            )

            train_stats = modules.train_early_stop(
                model=self._local_model, 
                trainloader=trainloader,
                testloader=testloader,
                epochs=self._trainer_epochs, 
                criterion=criterion,
                optimizer=optimizer,
                early_stop=self._early_stop,
                device=self._device
            )
            
            # Get public predictions
            distill_predicts = modules.predict_public(
                model=self._local_model,
                distill_loader=distillloader,
                predict_confid=self._predict_confid,
                onehot=self._onehot_output,
                device=self._device,
            )
        
        elif self._client_type == "heuristic":
            ## Heuristic Client Case - Random Predict
            
            t = 1000 * time.time() # current time in milliseconds
            np.random.seed(int(t) % 2**32)
            distill_predicts = np.random.randint(0, self._num_classes, len(self._distillset))
        
        elif self._client_type == "colluding":
            ## Colluding Client Case - Predict similar by colluding
            
            np.random.seed(self._random_seed)
            distill_predicts = np.random.randint(0, self._num_classes, len(self._distillset))

        ### Predictions Accuracy of the Worker
        print(f"Accuracy of disitllation set for Worker {self._client_id}: {np.count_nonzero(distill_predicts == distillloader.dataset.oTargets.detach().cpu().numpy()) / len(distillloader.dataset.oTargets)}")
        
        # Return the refined predictions
        predict_parameters = ndarrays_to_parameters([distill_predicts])
        fit_duration = timeit.default_timer() - fit_begin

        posttrain_loss, posttrain_accuracy = modules.evaluate(
            model=self._local_model, 
            testloader=testloader, 
            criterion=criterion,
            device=self._device
        )

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=predict_parameters,
            num_examples=len(trainloader),
            metrics={
                "client_id": self._client_id,
                "client_type": self._client_type,
                "fit_duration": fit_duration,
                "pr_accuracy": float(pretrain_accuracy),
                "pr_loss": float(pretrain_loss),
                "ps_accuracy": float(posttrain_accuracy),
                "ps_loss": float(posttrain_loss)
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.client_id}] evaluate, config: {ins.config}")
        
        criterion = modules.get_criterion(
            criterion_str=self._criterion_str
        )

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=self._batch_size, shuffle=False
        )
        loss, accuracy = modules.evaluate(
            model=self._local_model, 
            testloader=testloader, 
            criterion=criterion,
            device=self._device
        )
        
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(testloader),
            metrics={"accuracy": float(accuracy),
                     "loss": float(loss)},
        )
