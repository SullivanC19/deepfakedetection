import torch.nn as nn
from data.constants import IMAGE_SIZE
from .utils import Flatten

def cnn_model() -> nn.Module:
    return nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=7, padding=3),
           nn.ReLU(),
           nn.Dropout2d(p=0.2),
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Dropout2d(p=0.2),
           nn.MaxPool2d(kernel_size=4, stride=4),
           nn.Conv2d(64, 64, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Dropout2d(p=0.2),
           nn.Conv2d(64, 128, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Dropout2d(p=0.2),
           nn.MaxPool2d(kernel_size=4, stride=4),
           Flatten(),
           nn.Linear(128 * (IMAGE_SIZE // 16) ** 2, 1),
           nn.Sigmoid(),
        )