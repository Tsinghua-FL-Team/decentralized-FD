"""Training function to train the model for given number of epochs."""

import torch
import torch.nn as nn
from . import evaluate

def train(model,
          trainloader: torch.utils.data.DataLoader,
          epochs: int,
          device: str,  # pylint: disable=no-member
          criterion,
          optimizer,
         ) -> None:
    """Train the model."""
    # Define loss and optimizer
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    model.train()
    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 0 and i > 0:  # print every 500 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

def train_early_stop(
        model,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        criterion,
        optimizer,
        early_stop: float = -1,
        ) -> None:
    """Train the model."""
    # Define loss and optimizer
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    model.train()
    running_loss, samples = 0.0, 0
    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            samples += labels.shape[0]
            
            # print statistics
            running_loss += loss.item() * labels.shape[0]
            if i % 500 == 0 and i > 0:  # print every 500 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / samples))
            
            if early_stop != -1:
                # evaluate the performance of model
                _, accuracy = evaluate(
                    model=model, 
                    testloader=testloader, 
                    device=device
                )

                # do early stop if performance target achieved
                if accuracy >= early_stop:
                    print("Stopping criteria reached for worker...")
                    return

    train_stats = {"loss" : running_loss / samples}
        
    # return training statistics
    return train_stats

