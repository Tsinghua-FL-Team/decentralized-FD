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
    predictions = []
    with torch.no_grad():
        for data in distill_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = F.softmax(model(images), dim=1)
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
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()

    return predictions


def distill_train(
        model: nn.Module,
        distill_loader: torch.utils.data.DataLoader,
        num_classes: int,
        optimizer,
        distill_epochs: int,
        device: str,
    ) -> Dict[str, float]:
    
    running_loss, samples = 0.0, 0
    for epoch in range(distill_epochs):
        for x, y in distill_loader:   
            # create onehot encoding
            onehot_y = torch.zeros((len(y), num_classes))
            onehot_y[torch.arange(len(y)), y] = 1

            x, onehot_y = x.to(device), onehot_y.to(device)
            
            optimizer.zero_grad()
            y_ = F.softmax(model(x), dim = 1)
            
            # compute loss
            loss = nn.KLDivLoss(reduction="batchmean")(y_, onehot_y.detach())
            
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            
            loss.backward()
            optimizer.step()  

    distill_stats = {"loss" : running_loss / samples}

    return distill_stats

