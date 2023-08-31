"""A function to load the CIFAR-10 dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_stl10(data_root) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load CIFAR-10 (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Initialize Datasets. STL-10 will automatically download if not present
    trainset = torchvision.datasets.STL10(
        root=data_root, split='train', folds=None, download=True, transform=transform
    )
    testset = torchvision.datasets.STL10(
        root=data_root, split='test', folds=None, download=True, transform=transform
    )
    
    # Return the datasets
    return trainset, testset