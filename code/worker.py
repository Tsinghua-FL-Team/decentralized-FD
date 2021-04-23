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
#from torch.utils.data import DataLoader
#import copy

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
    def __init__(self, model_fn, optimizer_fn, tr_loader, counts, n_classes, 
                  ts_loader, ds_loader, early_stop=-1, init=None, idnum=None):
        self.id = idnum
        self.feature_extractor = None
        self.n_classes = n_classes
        self.early_stop = early_stop
        # local models and dataloaders
        self.tr_model = model_fn().to(device) #copy.deepcopy(model_fn()).to(device)
        self.tr_loader = tr_loader
        self.ts_loader = ts_loader
        self.ds_loader = ds_loader
        # optimizer parameters        
        self.optimizer_fn = optimizer_fn
        self.optimizer = optimizer_fn(self.tr_model.parameters())   
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)  
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
        loader = self.tr_loader if not loader else loader
        end_training = False
        itr = 0
        for ep in range(epochs):
            # train next epoch
            for i, x, y in loader:   
                itr += 1    
                x, y = x.to(device), y.to(device)
                
                self.optimizer.zero_grad()
                
                loss = nn.CrossEntropyLoss()(self.tr_model(x), y)
                
                running_loss += loss.item()*y.shape[0]
                samples += y.shape[0]
                
                loss.backward()
                self.optimizer.step()  

                # check for early stopping criteria
                if (self.early_stop != -1) and (itr % 5 == 0):
                    print("Checking early stop criteria...")
                    accuracy = self.evaluate()["accuracy"]
                    if accuracy >= self.early_stop:
                        print("Stopping criteria reached for worker {}...".format(self.id))
                        end_training = True
                        break
            # check if early stop criteria was reached
            if end_training:
                break
        train_stats = {"loss" : running_loss / samples}
        
        # return training statistics
        return train_stats


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Evaluate Worker to see if it is improving as expected or not.     #
    #                                                                     #
    #---------------------------------------------------------------------#
    def evaluate(self, ts_loader=None):
        """Evaluation function to check performance"""
        # start evaluation of the model
        self.tr_model.eval()
        samples, correct = 0, 0
        
        # check if a dataloader was provided for evaluation
        loader = self.ts_loader if not ts_loader else ts_loader
        
        with torch.no_grad():
            for x, y in loader:
                
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
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Prediction functions to make predictions on public dataset.       #
    #                                                                     #
    #---------------------------------------------------------------------#
    def predict_public(self, ds_loader=None, use_confid=False, confid=None):
        if use_confid:
            self.distill_predict_with_confid(ds_loader=ds_loader, 
                                             confidence=confid)
        else:
            self.distill_predict(ds_loader=ds_loader)

    def distill_predict(self, ds_loader=None):
        
        # check if a dataloader was provided
        loader = self.ds_loader if not ds_loader else ds_loader

        predictions = []

        # compute predictions
        for x, _ in loader:
            x = x.to(device)
            #predictions += [self.predict_max(x).detach()]
            # get prediction
            y_ = self.predict(x)

            # find agrument max
            amax = torch.argmax(y_, dim=1).detach()
            predictions += [amax]

        # collect results to cpu  memory and numpy arrays
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy().astype("uint8")
        
        print((predictions == loader.dataset.oTargets).sum())
        
        # append past resutls
        self.predictions.append(predictions)
                #np.argmax(predictions, axis=-1).astype("uint8")

    def distill_predict_with_confid(self, confidence, ds_loader=None):
        
        # check if a dataloader was provided
        loader = self.ds_loader if not ds_loader else ds_loader
        
        predictions = []

        # compute predictions
        for x, _ in loader:
            x = x.to(device)

            # get prediction
            y_ = self.predict(x)

            # find agrument max
            amax = torch.argmax(y_, dim=1).detach()

            # get prediction confidence
            pred_confid = y_[np.arange(y_.shape[0]), amax]

            # apply the confidence threshold for predictions
            amax[pred_confid < confidence] = -1
            predictions += [amax]

        # collect results to cpu  memory and numpy arrays
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        
        # append past resutls
        self.predictions.append(predictions)
        
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used by workers to communicate with server.             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def get_from_server (self, server):
        # fetch global aggregated labels from server / contract
        self.global_labels =  server.global_labels()


    def send_to_server (self, server):

        # compute label distribution in my predictions
        label_distribution = [np.count_nonzero(self.predictions[-1] == c) for c in range(self.n_classes)]

        # send both predictions and distribution to the server
        server.receive_prediction(
            w_id=self.id, 
            predictions=self.predictions[-1], 
            freq=label_distribution)
    
    
    #---------------------------------------------------------------------#
    #                                                                     #
    #   Performs federated distillation step.                             #
    #                                                                     #
    #---------------------------------------------------------------------#
    def distill(self, distill_iter, ds_loader=None, reset_optimizer=False):
        """Distillation function to perform Federated Distillation"""
        print("Distilling on Worker {}...".format(self.id))
        
        if reset_optimizer:
            self.optimizer = self.optimizer_fn(self.tr_model.parameters())
            print("optimizer reset...")

        # check if a dataloader was provided
        loader = self.ds_loader if not ds_loader else ds_loader
        
        # start training the worker using distill dataset
        self.tr_model.train()  
        
        # setup global labels
        loader.dataset.setTargets(labels=self.global_labels)
        
        print(np.count_nonzero(self.global_labels == loader.dataset.oTargets))
        print(np.count_nonzero(self.predictions[-1] == loader.dataset.oTargets))
        
        itr = 0
        while True:
            running_loss, samples = 0.0, 0
            for x, y in loader:   
                itr += 1
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

            if itr >= distill_iter:
                distill_stats = {"loss" : running_loss / samples}

        # return distillation statistics
        return distill_stats
