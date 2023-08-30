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
            model_fn: torch.nn.Module,
            trainset: Dataset,
            testset: Dataset,
            distillset: Dataset,
            early_stop: float,
            predict_confid: float,
            criterion_str: str,
            optimizer_str: str,
            num_classes: int,
            co_distill_epochs: int,
            distill_epochs: int,
            trainer_epochs: int,
            learning_rate: float,
            batch_size: int,
            client_type: str,
            random_seed: int,
            onehot_output: bool,
            device: str,
            reset_model: bool = False,
            reset_optim: bool = False,
            co_distill: bool = True,
            ) -> None:
        """Initializes a new honest client."""
        super().__init__()
        self.client_id = client_id
        self.model_fn = model_fn
        self.trainset = trainset
        self.testset = testset
        self.distillset = distillset
        self.early_stop = early_stop
        self.predict_confid = predict_confid
        self.optimizer_str = optimizer_str
        self.criterion_str = criterion_str
        self.num_classes = num_classes
        self.co_distill_epochs = co_distill_epochs
        self.distill_epochs = distill_epochs
        self.trainer_epochs = trainer_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.client_type = client_type
        self.random_seed = random_seed
        self.onehot_output = onehot_output
        self.reset_model = reset_model
        self.reset_optim = reset_optim
        self.co_distill = co_distill
        self.device = device

        # Initialize models
        torch.manual_seed(self.random_seed)
        self.train_model = self.model_fn(num_classes=self.num_classes).to(self.device)
        self.distill_model = self.model_fn(num_classes=self.num_classes).to(self.device)
        self.co_distill_model = None

        # Setup Optimizers
        self.train_optimizer = modules.get_optimizer(
            optimizer_str=self.optimizer_str,            
            local_model=self.train_model,
            learning_rate=self.learning_rate,
        )
        self.distill_optimizer = modules.get_optimizer(
            optimizer_str=self.optimizer_str,            
            local_model=self.distill_model,
            learning_rate=self.learning_rate,
        )
        self.co_distill_optimizer = None
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Module to fetch model parameters of current client."""
        print(f"[Client {self.client_id}] get_parameters, config: {ins.config}")

        weights = self.train_model.get_weights()
        parameters = ndarrays_to_parameters(weights)
        
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )

    def distill(self, global_labels):
        self.distillset.setTargets(labels=global_labels)

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
            self.trainset, batch_size=self.batch_size, shuffle=True
        )
        
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False
        )        
        
        distillloader = torch.utils.data.DataLoader(
            self.distillset, batch_size=self.batch_size, shuffle=False
        )        
            
        criterion = modules.get_criterion(
            criterion_str=self.criterion_str
        )
        
        if self.reset_optim:
            self.train_optimizer = modules.get_optimizer(
                optimizer_str=self.optimizer_str,            
                local_model=self.train_model,
                learning_rate=self.learning_rate,
            )
            self.distill_optimizer = modules.get_optimizer(
                optimizer_str=self.optimizer_str,
                local_model=self.distill_model,
                learning_rate=self.learning_rate,
            )
        
        
        distill_stats, co_distill_stats, train_stats = None, None, None
        distill_loss, distill_accuracy = 0.0, 0.0
        co_distill_loss, co_distill_accuracy = 0.0, 0.0
        train_loss, train_accuracy = 0.0, 0.0
        public_predicts = None

        if self.client_type == "normal":
            ## Normal Client Case - Distill -> Train -> Predict
            public_labels = parameters_to_ndarrays(ins.parameters)
            if len(public_labels[0]) > 0:

                # Initialize distillation model if this
                # is the first run of model training
                if self.reset_model:
                    torch.manual_seed(self.random_seed)
                    self.distill_model = self.model_fn(num_classes=self.num_classes).to(self.device)
                
                # Assign public labels to the distill dataset
                distillloader.dataset.setTargets(labels=public_labels[0][0])
                distill_stats = modules.distill_train(
                    model=self.distill_model,
                    distill_loader=distillloader,
                    optimizer=self.distill_optimizer,
                    device=self.device,
                    num_classes=self.num_classes,
                    distill_epochs=self.distill_epochs,
                )
                ### Predictions Accuracy of the Global Labels
                if self.client_id == 0:
                    print(f"Global Accuracy of the distillation set: {np.count_nonzero(distillloader.dataset.targets == distillloader.dataset.oTargets) / len(distillloader.dataset.oTargets)}")


                if self.co_distill and self.distill_model is not None:
                    self.co_distill_model = self.model_fn(num_classes=self.num_classes).to(self.device)
                    self.co_distill_optimizer = modules.get_optimizer(
                        optimizer_str=self.optimizer_str,
                        local_model=self.co_distill_model,
                        learning_rate=self.learning_rate,
                    )
                    # Get distill predictions
                    # distill_predicts = modules.predict_public(
                    #     model=self.distill_model,
                    #     distill_loader=distillloader,
                    #     predict_confid=0.0,
                    #     onehot=True,
                    #     device=self.device,
                    # )
                    # Setup distill predictions
                    # distill_targets = torch.argmax(distill_predicts, dim=1).detach().cpu().numpy()
                    # distillloader.dataset.setTargets(labels=distill_targets)
                    # Train co-distill model
                    co_distill_stats = modules.co_distill_train(
                        co_distill_model=self.co_distill_model,
                        distill_model=self.distill_model,
                        distill_loader=distillloader,
                        optimizer=self.co_distill_optimizer,
                        device=self.device,
                        num_classes=self.num_classes,
                        distill_epochs=self.co_distill_epochs,
                    )

            distill_loss, distill_accuracy = modules.evaluate(
                model=self.distill_model, 
                testloader=testloader, 
                criterion=criterion,
                device=self.device
            )
            
            co_distill_loss, co_distill_accuracy = modules.evaluate(
                model=self.co_distill_model, 
                testloader=testloader, 
                criterion=criterion,
                device=self.device
            )

            # Setup the training model for
            # training with local dataset
            if self.co_distill_model is None:
                distill_state = self.distill_model.state_dict()
            else:
                distill_state = self.co_distill_model.state_dict()

            distill_state = {k : v for k, v in distill_state.items() if "binary" not in k}
            self.train_model.load_state_dict(distill_state, strict=False)


            # Finally train the client side model
            # using the local training data
            train_stats = modules.train_early_stop(
                model=self.train_model, 
                trainloader=trainloader,
                testloader=testloader,
                epochs=self.trainer_epochs, 
                criterion=criterion,
                optimizer=self.train_optimizer,
                early_stop=self.early_stop,
                device=self.device
            )
            
            # Get public predictions
            public_predicts = modules.predict_public(
                model=self.train_model,
                distill_loader=distillloader,
                predict_confid=self.predict_confid,
                onehot=self.onehot_output,
                device=self.device,
            )
            public_predicts = public_predicts.detach().cpu().numpy()
        
        elif self.client_type == "heuristic":
            ## Heuristic Client Case - Random Predict
            
            t = 1000 * time.time() # current time in milliseconds
            np.random.seed(int(t) % 2**32)
            public_predicts = np.random.randint(0, self.num_classes, len(self.distillset))
        
        elif self.client_type == "colluding":
            ## Colluding Client Case - Predict similar by colluding
            
            np.random.seed(self.random_seed)
            public_predicts = np.random.randint(0, self.num_classes, len(self.distillset))

        ### Predictions Accuracy of the Worker
        print(f"Accuracy of disitllation set for Worker {self.client_id}: {np.count_nonzero(np.argmax(public_predicts, axis=-1) == distillloader.dataset.oTargets.detach().cpu().numpy()) / len(distillloader.dataset)}")
        
        # Return the refined predictions
        predict_parameters = ndarrays_to_parameters([public_predicts])
        fit_duration = timeit.default_timer() - fit_begin

        train_loss, train_accuracy = modules.evaluate(
            model=self.train_model, 
            testloader=testloader, 
            criterion=criterion,
            device=self.device
        )

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=predict_parameters,
            num_examples=len(trainloader),
            metrics={
                "client_id": self.client_id,
                "client_type": self.client_type,
                "fit_duration": fit_duration,
                # "distill_stats": distill_stats,
                # "co_distill_stats": co_distill_stats,
                # "train_stats": train_stats,
                "distill_loss": float(distill_loss),
                "distill_accuracy": float(distill_accuracy),
                "codistill_loss": float(co_distill_loss),
                "codistill_accuracy": float(co_distill_accuracy),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.client_id}] evaluate, config: {ins.config}")
        
        criterion = modules.get_criterion(
            criterion_str=self.criterion_str
        )

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False
        )
        loss, accuracy = modules.evaluate(
            model=self.train_model, 
            testloader=testloader, 
            criterion=criterion,
            device=self.device
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
