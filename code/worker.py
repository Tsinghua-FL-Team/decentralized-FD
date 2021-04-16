#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy

#-----------------------------------------------------------------------------#
#                                                                             #
#   Define global parameters to be used through out the program               #
#                                                                             #
#-----------------------------------------------------------------------------#
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   class that implements worker node logic for Federated Distillation.       #
#                                                                             #
#*****************************************************************************#
class Worker():
    def __init__(self, model_fn, optimizer_fn, loader, counts, n_classes, init=None, idnum=None):
        self.id = idnum
        self.feature_extractor = None
        #self.distill_loader = distill_loader
        self.n_classes = n_classes
        # model parameters
        self.tr_model = model_fn().to(device) #copy.deepcopy(model_fn()).to(device) #(nn.Module) 
        #self.distill_model = copy.deepcopy(model_fn).to(device)
        self.loader = loader
        #self.W = {key : value for key, value in self.tr_model.named_parameters()}
        #self.dW = {key : torch.zeros_like(value) for key, value in self.tr_model.named_parameters()}
        #self.W_old = {key : torch.zeros_like(value) for key, value in self.tr_model.named_parameters()}
        # optimizer parameters        
        self.optimizer_fn = optimizer_fn
        self.optimizer = optimizer_fn(self.tr_model.parameters())   
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)  
        #self.c_round = 0
        # result variables
        self.predictions = []

    #---------------------------------------------------------------------#
    #                                                                     #
    #   Train Worker using its local dataset.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def train(self, epochs=1, loader=None, reset_optimizer=False):
        """Training function to train local model"""
        if reset_optimizer:
            self.optimizer = self.optimizer_fn(self.tr_model.parameters())  
        
        # start training the worker using local dataset
        self.tr_model.train()  
        running_loss, samples = 0.0, 0
        
        # check if a dataloader was provided for training
        loader = self.loader if not loader else loader
        
        for ep in range(epochs):
            for i, x, y in loader:   

                x, y = x.to(device), y.to(device)
                
                self.optimizer.zero_grad()
                
                loss = nn.CrossEntropyLoss()(self.tr_model(x), y)

                #if lambda_fedprox != 0.0:
                #  loss += lambda_fedprox * torch.sum((flatten(W0).cuda()-flatten(dict(model.named_parameters())).cuda())**2)
                
                running_loss += loss.item()*y.shape[0]
                samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()  
                #scheduler.step()

        train_stats = {"loss" : running_loss / samples}
        #print(self.label_counts)
        #eval_scores(self.tr_model, self.distill_loader)
          
        return train_stats
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Evaluate Worker to see if it is improving as expected or not.     #
    #                                                                     #
    #---------------------------------------------------------------------#
    def evaluate(self, loader=None):
        """Evaluation function to check performance"""
        # start evaluation of the model
        self.tr_model.eval()
        samples, correct = 0, 0
        
        # check if a dataloader was provided for evaluation
        loader = self.loader if not loader else loader
        
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                
                x, y = x.to(device), y.to(device)
                
                y_ = self.tr_model(x)
                _, predicted = torch.max(y_.detach(), 1)
                
                samples += y.shape[0]
                correct += (predicted == y).sum().item()
        
        # return evaluation statistics
        return {"accuracy" : correct/samples}


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Prediction functions used by the worker.                          #
    #                                                                     #
    #---------------------------------------------------------------------#
    def predict_logit(self, x):
        """Logit prediction on input"""
        with torch.no_grad():
            y_ = self.tr_model(x)
        return y_

    def predict(self, x):
        """Softmax prediction on input"""
        with torch.no_grad():
            y_ = F.softmax(self.predict_logit(x), dim = 1)
        return y_

    def predict_max(self, x):
        """Onehot Argmax prediction on input"""
        y_ = self.predict(x)
        amax = torch.argmax(y_, dim=1).detach()
        t = torch.zeros_like(y_)
        t[torch.arange(y_.shape[0]),amax] = 1
        return t
    
    def compute_distill_predictions(self, loader=None):
        
        # check if a dataloader was provided
        loader = self.loader if not loader else loader
        
        predictions = []

        # compute predictions
        for x, _ in loader:
            x = x.to(device)
            predictions += [self.predict_max(x).detach()]
        
        # collect results to cpu  memory and numpy arrays
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        
        # append past resutls
        self.predictions.append(
            np.argmax(predictions, axis=-1).astype("uint8")
        )


        
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used by workers to communicate with server.             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def get_from_server (self, server):
        self.global_labels =  server.global_labels()
        
        # get onehot encoding of the same        
        #self.onehot_distill_labels = np.zeros(
        #    (distill_labels.size, self.n_classes), dtype="long")
        
        #self.onehot_distill_labels[
        #    np.arange(distill_labels.size),distill_labels] = 1
        

    def send_to_server (self, server):
        # compute label distribution in my predictions
        label_distribution = [
            np.count_nonzero(self.predictions[-1] == c) 
            for c in range(self.n_classes)
            ]
        #np.unique(self.predictions, return_counts=True)
        # send both predictions and distribution to the server
        server.receive_prediction(
            w_id=self.id, 
            predictions=self.predictions[-1], 
            freq=label_distribution
        )
    
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Performs federated distillation step.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def distill(self, distill_epochs, loader=None, reset_optimizer=False):
        """Distillation function to perform Federated Distillation"""
        if reset_optimizer:
            self.optimizer = self.optimizer_fn(self.tr_model.parameters())  

        # check if a dataloader was provided
        loader = self.loader if not loader else loader
        
        # start training the worker using distill dataset
        self.tr_model.train()  
        running_loss, samples = 0.0, 0

        # setup global labels
        loader.dataset.setTargets(labels=self.global_labels)
        
        for ep in range(distill_epochs):
            for x, y in loader:   
                # create onehot encoding
                onehot_y = torch.zeros((len(y), self.n_classes))
                onehot_y[torch.arange(len(y)), y] = 1

                x, onehot_y = x.to(device), onehot_y.to(device)
                
                self.optimizer.zero_grad()
                y_ = F.softmax(self.tr_model(x), dim=1)
                
                # compute loss
                loss = nn.KLDivLoss(reduction="batchmean")(y_, onehot_y.detach())
                
                running_loss += loss.item() * y.shape[0]
                samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()  

        distill_stats = {"loss" : running_loss / samples}
          
        return distill_stats
