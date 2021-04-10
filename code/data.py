#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch, torchvision
import torchvision.transforms as transforms
import numpy as np


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load and return the training and testing sets of CIFAR-10 dataset.        #
#                                                                             #
#*****************************************************************************#
def load_cifar10(path):
    """Load CIFAR-10 (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
        ])
    
    # Initialize Datasets. CIFAR-10 will automatically download if not present
    trainset = torchvision.datasets.CIFAR10(
        root=path+"CIFAR10", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=path+"CIFAR10", train=False, download=True, transform=transform
    )
    
    # Return the datasets
    return trainset, testset

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load and return the training and testing sets of CIFAR-100 dataset.       #
#                                                                             #
#*****************************************************************************#
def load_cifar100(path):
    """Load CIFAR-100 (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
        ])
    
    # Initialize Datasets. CIFAR-10 will automatically download if not present
    trainset = torchvision.datasets.CIFAR10(
        root=path+"CIFAR100", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=path+"CIFAR100", train=False, download=True, transform=transform
    )
    
    # Return the datasets
    return trainset, testset

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load and return the training and testing sets of MNIST dataset.           #
#                                                                             #
#*****************************************************************************#
def load_mnist(path):
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([torchvision.transforms.Resize((32,32)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.MNIST(
        root=path+"MNIST", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=path+"MNIST", train=False, download=True, transform=transform
    )

    # Return the datasets
    return trainset, testset

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load and return the training and testing sets of EMNIST dataset.          #
#                                                                             #
#*****************************************************************************#
def load_emnist(path):
    """Load EMNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([torchvision.transforms.Resize((32,32)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.EMNIST(
        root=path+"EMNIST", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=path+"EMNIST", train=False, download=True, transform=transform
    )

    # Return the datasets
    return trainset, testset

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load and return the training and testing sets of STL-10 dataset.          #
#                                                                             #
#*****************************************************************************#
def load_stl10(path):
    """Load STL-10 (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([torchvision.transforms.Resize((32,32)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(
                                    (0.4914, 0.4822, 0.4465), 
                                    (0.2023, 0.1994, 0.2010)
                                    )
                                   ])

    # Initialize Datasets. MNIST will automatically download if not present
    dataset = torchvision.datasets.STL10(
        root=path+"STL10", download=True, transform=transform
    )
    #, split="train"

    # Return the datasets
    return dataset

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   split given dataset among workers using dirichlet distribution.           #
#                                                                             #
#*****************************************************************************#
def split_dirichlet(labels, n_workers, n_data, alpha, double_stochstic=True):
    """Splits data among the workers using dirichlet distribution"""

    #np.random.seed(0)

    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()
    
    n_classes = np.max(labels)+1
    
#    if alpha == 0:
#        alpha = n_classes
    label_distribution = np.random.dirichlet([alpha]*n_workers, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]

    worker_idcs = [[] for _ in range(n_workers)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            worker_idcs[i] += [idcs]

    worker_idcs = [np.concatenate(idcs) for idcs in worker_idcs]

    #print_split(worker_idcs, labels)
  
    return worker_idcs

def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    #x = x / x.sum(axis=0).reshape(1,-1)
    return x

class IdxSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return (*self.dataset[self.indices[idx]], idx)

    def __len__(self):
        return len(self.indices)

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   split and return datasets for workers.                                    #
#                                                                             #
#*****************************************************************************#
def split_data(train_data, n_workers=10, classes_per_worker=0, n_data=None):
    """Split data among Worker nodes."""
    
    # Find allocated indices using dirichlet split
    subset_idx = split_dirichlet(
        train_data.targets, 
        n_workers, 
        n_data, 
        classes_per_worker
        )
    
    # Compute labels per worker
    label_counts = [np.bincount(np.array(train_data.targets)[i], minlength=10) 
                    for i in subset_idx]
    
    # Get actual worker data
    worker_data = [IdxSubset(train_data, subset_idx[i]) 
                   for i in range(n_workers)]

    # Return worker data splits
    return worker_data, label_counts

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   load dataset as required.                                                 #
#                                                                             #
#*****************************************************************************#
def load_data(dataset, path):
  return {"cifar10" : load_cifar10 , 
          "mnist" : load_mnist , 
          "emnist" : load_emnist ,
          "stl10" : load_stl10 , 
          "cifar100" : load_cifar100
          }[dataset](path)


