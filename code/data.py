#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch, torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split as splitter

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
#   a class to create custom subsets with indexes appended.                   #
#                                                                             #
#*****************************************************************************#
class IdxSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return (idx, *self.dataset[self.indices[idx]])

    def __len__(self):
        return len(self.indices)


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   a class to create custom subsets, should be used to change all datasets.  #
#                                                                             #
#*****************************************************************************#
class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. 
                                will be the same length as indices
    """
    def __init__(self, dataset, indices, labels=None):
        self.dataset = dataset
        self.indices = indices
        if not labels:
            targets = np.array(self.dataset.targets)[indices]
            self.targets = torch.tensor(targets).long()
            # original labels
            self.oTargets = targets
        else:
            self.targets = torch.tensor(labels).long()
   
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]][0]
        target = self.targets[idx]
        return (data, target)

    def __len__(self):
        return len(self.targets)
    
    def setTargets(self, labels):
        self.targets = torch.tensor(labels).long()        


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   make uneven splits of given dataset for each worker as defined.           #
#                                                                             #
#*****************************************************************************#
def uneven_split(labels, n_workers, n_data, classes_per_worker):
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels) + 1
    label_idcs = {l : np.argwhere(np.array(labels)==l).flatten().tolist() for l in range(n_classes)}
    #class_idcs = [np.argwhere(np.array(labels)==y).flatten() for y in range(n_classes)]
    
    classes_per_worker = n_classes if classes_per_worker == 0 else classes_per_worker

    idcs = []
    for i in range(n_workers):
        worker_idcs = []
        budget = n_data[i]
        c = np.random.randint(n_classes)
        while budget > 0:
            take = min(n_data[i] // classes_per_worker, len(label_idcs[c]), budget)
            
            worker_idcs += label_idcs[c][:take]
            label_idcs[c] = label_idcs[c][take:]
            
            budget -= take
            c = (c + 1) % n_classes
        idcs += [worker_idcs]

    
    print_split(idcs, labels)
    
    return idcs


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   split given dataset among workers using dirichlet distribution.           #
#                                                                             #
#*****************************************************************************#
def split_dirichlet(labels, n_workers, alpha, double_stochstic=True):
    """Splits data among the workers using dirichlet distribution"""

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels)+1
    
    # get label distibution
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

    print_split(worker_idcs, labels)
  
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


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   split and return datasets for workers.                                    #
#                                                                             #
#*****************************************************************************#
def split_data(train_data, alpha, n_workers=10, worker_data=None, classes_per_worker=None):
    """Split data among Worker nodes."""
    
    # Find allocated indices using dirichlet split
    if not worker_data:
        subset_idx = split_dirichlet(train_data.targets, n_workers, alpha)
    else:
        subset_idx = uneven_split(train_data.targets, n_workers, worker_data, classes_per_worker)
    
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
#   split given dataset to create a test set and a ditillation set.           #
#                                                                             #
#*****************************************************************************#
def create_distill(dataset, random_seed, n_distill):
    """Split dataset into test and distill set according to given ratio."""
    distill_portion = float(n_distill / len(dataset.targets))
    
    # Get index for distill and test datasets
    test_idx, distill_idx = splitter(
        np.arange(len(dataset.targets)), 
        test_size=distill_portion,
        shuffle=True,
        stratify=dataset.targets, 
        random_state=random_seed
    )
    
    # create actual subsets
    test_set = CustomSubset(dataset, test_idx)
    distill_set = CustomSubset(dataset, distill_idx)

    return test_set, distill_set
    
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


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   helper function to print data splits made for workers.                    #
#                                                                             #
#*****************************************************************************#
def print_split(idcs, labels):
    
    n_labels = np.max(labels) + 1 
    print("Data split:")
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
        splits += [split]
        if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
            print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)
    
    print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
    print()
