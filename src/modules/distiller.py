"""Evaluation function to test the model performance."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def predict_public(
        model: nn.Module,
        distill_loader: torch.utils.data.DataLoader,
        predict_confid: float,
        device: str,
        onehot: bool = False,
    ) -> List[int]:
    
    """Validate the model on the entire test set."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in distill_loader:
            x, y = x.to(device), y.to(device)
            outputs = F.softmax(model(x), dim=1)
            # return probabilities or one-hot 
            # encoded class predictions
            if onehot:
                # create one hot encoding for highest
                # probability class and apply confidence
                # check on it.
                amax = torch.argmax(outputs, dim=1).detach()
                onehot_encoded = F.one_hot(amax, num_classes=10)
                pred_conf = outputs[np.arange(outputs.shape[0]), amax]
                onehot_encoded[pred_conf < predict_confid, :] = 0
                predictions += [onehot_encoded]
            else:
                # Simply collect prediction confidence
                predictions += [outputs]

        # Collect all results and convert to numpy arrays
        predictions = torch.cat(predictions, dim=0).detach()

    return predictions


def distill_train(
        distill_model: nn.Module,
        distill_loader: torch.utils.data.DataLoader,
        num_classes: int,
        optimizer,
        distill_epochs: int,
        device: str,
    ) -> Dict[str, float]:
    
    distill_model.train()
    running_loss, samples = 0.0, 0
    for epoch in range(distill_epochs):
        for x, y in distill_loader:   
            # create onehot encoding
            onehot_y = torch.zeros((len(y), num_classes))
            onehot_y[torch.arange(len(y)), y] = 1

            x, onehot_y = x.to(device), onehot_y.to(device)
            
            optimizer.zero_grad()
            y_ = F.softmax(distill_model(x), dim = 1)
            
            # compute loss
            loss = kulbach_leibler_divergence(y_, onehot_y.detach())
            #loss = nn.KLDivLoss(reduction="batchmean")(y_, onehot_y.detach())
            
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            
            loss.backward()
            optimizer.step()  

    distill_stats = {"loss" : running_loss / samples}

    return distill_stats


def co_distill_train(
        distill_model: nn.Module,
        co_distill_model: nn.Module,
        distill_loader: torch.utils.data.DataLoader,
        num_classes: int,
        optimizer,
        distill_epochs: int,
        device: str,
    ) -> Dict[str, float]:
    
    distill_model.train()
    co_distill_model.train()
    
    running_loss, samples = 0.0, 0
    for epoch in range(distill_epochs):
        for x, _ in distill_loader:   
            x = x.to(device)
            y = F.softmax(distill_model(x), dim = 1)
            
            optimizer.zero_grad()
            y_ = F.softmax(co_distill_model(x), dim = 1)
            
            # compute loss
            loss = kulbach_leibler_divergence(y_, y.detach())
            # loss = nn.KLDivLoss(reduction="batchmean")(y_, y)
            
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            
            loss.backward()
            optimizer.step()  

    distill_stats = {"loss" : running_loss / samples}

    return distill_stats


# Custom KL-Divergence Code
def kulbach_leibler_divergence(predicted, target):
  return -(target * torch.log(predicted.clamp_min(1e-7))).sum(dim=-1).mean() 
