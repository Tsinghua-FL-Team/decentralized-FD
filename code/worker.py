#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   class that implements worker node logic for Federated Distillation.       #
#                                                                             #
#*****************************************************************************#
class Worker():
    def __init__(self, model_fn, optimizer_fn, loader, counts, 
                 distill_loader, device="cpu", init=None, idnum=None):
        #super().__init__(model_fn, optimizer_fn, loader, init)
        self.id = idnum
        self.feature_extractor = None
        self.distill_loader = distill_loader
        self.device = device
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
    def train(self, epochs=1, loader=None, reset_optimizer=False, **kwargs):
        
        if reset_optimizer:
            self.optimizer = self.optimizer_fn(self.model.parameters())  
        
        # start training the worker using local dataset
        self.model.train()  
        running_loss, samples = 0.0, 0
        
        for ep in range(epochs):
            for x, y, source, index in loader:   
                x, y = x.to(self.device), y.to(self.device)
                
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
            y_ = nn.softmax(1)(self.predict_logits(x))
        return y_
    
    def compute_prediction_matrix(self, argmax=True):
        predictions = []
        idcs = []
        for x, _, idx in self.distill_loader:
            x = x.to(self.device)
            s_predict = self.predict(x).detach()
            predictions += [s_predict]
            idcs += [idx]
        
        argidx = torch.argsort(torch.cat(idcs, dim=0))
        predictions =  torch.cat(predictions, dim=0)[argidx].detach().cpu().numpy()
        
        if argmax:
            return np.argmax(predictions, axis=-1).astype("uint8")
        else:
            return predictions.astype("float16")
        
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Performs federated distillation step.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def distillation(self):
        return 0