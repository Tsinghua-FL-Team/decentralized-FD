#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.model = model_fn().to(device)
        self.loader = loader
        self.W = {key : value for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        # optimizer parameters        
        self.optimizer_fn = optimizer_fn
        self.optimizer = optimizer_fn(self.model.parameters())   
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)  
        self.c_round = 0

    #---------------------------------------------------------------------#
    #                                                                     #
    #   Train Worker using its local dataset.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def train(self, epochs=1, loader=None, reset_optimizer=False):
        """Training function to train local model"""
        if reset_optimizer:
            self.optimizer = self.optimizer_fn(self.model.parameters())  
        
        # start training the worker using local dataset
        self.model.train()  
        running_loss, samples = 0.0, 0
        
        # check if a dataloader was provided for training
        loader = self.loader if not loader else loader
        
        for ep in range(epochs):
            for i, x, y in loader:   

                x, y = x.to(device), y.to(device)
                
                self.optimizer.zero_grad()
                
                loss = nn.CrossEntropyLoss()(self.model(x), y)
                
                #if lambda_fedprox != 0.0:
                #  loss += lambda_fedprox * torch.sum((flatten(W0).cuda()-flatten(dict(model.named_parameters())).cuda())**2)
                
                running_loss += loss.item()*y.shape[0]
                samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()  
                #scheduler.step()

        train_stats = {"loss" : running_loss / samples}
        #print(self.label_counts)
        #eval_scores(self.model, self.distill_loader)
          
        return train_stats
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Evaluate Worker to see if it is improving as expected or not.     #
    #                                                                     #
    #---------------------------------------------------------------------#
    def evaluate(self, loader=None):
        """Evaluation function to check performance"""
        # start evaluation of the model
        self.model.eval()
        samples, correct = 0, 0
        
        # check if a dataloader was provided for evaluation
        loader = self.loader if not loader else loader
        
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                
                x, y = x.to(device), y.to(device)
                
                y_ = self.model(x)
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
            y_ = self.model(x)
        return y_

    def predict(self, x):
        """Softmax prediction on input"""
        with torch.no_grad():
            y_ = F.softmax(self.predict_logit(x), dim = 1)
        return y_
    
    def compute_prediction_matrix(self, loader=None, argmax=True):
        predictions = []
        idcs = []
        
        # check if a dataloader was provided
        loader = self.loader if not loader else loader
        
        for x, idx in loader:
            x = x.to(device)
            s_predict = self.predict(x).detach()
            predictions += [s_predict]
            idcs += [idx]
        
        argidx = torch.argsort(torch.cat(idcs, dim=0))
        predictions =  torch.cat(predictions, dim=0)[argidx].detach().cpu().numpy()
        
        if argmax:
            predictions = np.argmax(predictions, axis=-1).astype("uint8")
        else:
           predictions = predictions.astype("float16")
        
        self.predictions = predictions

        
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used by workers to communicate with server.             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def get_from_server (self, server):
        self.distill_labels = server.global_labels()

    def send_to_server (self, server):
        # compute label distribution in my predictions
        label_distribution = [np.count_nonzero(self.predictions == c) 
                              for c in range(self.n_classes)
                             ]
        #np.unique(self.predictions, return_counts=True)
        # send both predictions and distribution to the server
        server.receive_prediction(w_id=self.id, predictions=self.predictions, freq=label_distribution)
    
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Performs federated distillation step.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def distillation(self, loader=None):
        # check if a dataloader was provided
        loader = self.loader if not loader else loader
        
        return 0