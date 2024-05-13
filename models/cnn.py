import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def model(in_channel, channel_1, channel_2, channel_3, channel_4, img_size, num_classes):
    return nn.Sequential(
           nn.Conv2d(in_channel, channel_1, 5, padding=2, padding_mode='zeros', bias=True),
           nn.BatchNorm2d(channel_1),
           nn.ReLU(),
           nn.Conv2d(channel_1, channel_2, 7, padding=3, padding_mode='zeros', bias=True),
           nn.BatchNorm2d(channel_2),
           nn.ReLU(),
           nn.Conv2d(channel_2, channel_3, 7, padding=3, padding_mode='zeros', bias=True),
           nn.BatchNorm2d(channel_3),
           nn.ReLU(),
           nn.Conv2d(channel_3, channel_4, 5, padding=2, padding_mode='zeros', bias=True),
           nn.BatchNorm2d(channel_4),
           nn.ReLU(),
           Flatten(),
           nn.Linear(channel_4*img_size*img_size, num_classes),
           nn.Sigmoid()
        )