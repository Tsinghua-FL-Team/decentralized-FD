"""Evaluation function to test the model performance."""

from typing import Tuple

import torch
import torch.nn as nn

def evaluate(
        model,
        testloader: torch.utils.data.DataLoader,
        criterion,
        device: str,
    ) -> Tuple[float, float]:
    
    """Validate the model on the entire test set."""
    if model is None:
        return 0.0, 0.0

    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
            loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs.detach(), 1)  
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = correct / total
    
    return loss, accuracy