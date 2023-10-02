"""Implementation of a simple multilayer perceptron network."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import SqueezeNet

import flwr as fl

class Net(SqueezeNet):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__(num_classes=num_classes)
        self.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2)
        self._num_classes = num_classes

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

if __name__=="__main__":
    model = Net(10)
        