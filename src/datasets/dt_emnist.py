"""A function to load the MNIST digit dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_emnist(data_root, split="digits") -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),    
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.EMNIST(
        root=data_root, train=True, split=split, download=True, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=data_root, train=False, split=split, download=True, transform=transform
    )

    # Return the datasets
    return trainset, testset