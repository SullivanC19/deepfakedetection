import torch.nn as nn

from .utils import Flatten
from data.constants import IMAGE_SIZE

def lin_model() -> nn.Module:
    return nn.Sequential(
        Flatten(),
        nn.Linear(3 * IMAGE_SIZE ** 2, 1),
        nn.Sigmoid(),
    )
