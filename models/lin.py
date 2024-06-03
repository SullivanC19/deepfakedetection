import torch.nn as nn

from utils import Flatten
from data.constants import IMAGE_SIZE

def lin_model() -> nn.Module:
    return nn.Sequential(
        Flatten(),
        nn.Linear(IMAGE_SIZE ** 2, 10),
        nn.Sigmoid(),
    )
