import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        print(x.shape)
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def cnn_model(channels, img_dim):
    return nn.Sequential(
           nn.Conv2d(channels[0], channels[1], kernel_size=7, stride=3, padding=3, padding_mode='zeros', bias=True),
           nn.BatchNorm2d(channels[1]),
           nn.ReLU(),
           nn.Conv2d(channels[1], channels[2], kernel_size=7, stride=3, padding=3, padding_mode='zeros', bias=True),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.Conv2d(channels[2], channels[3], kernel_size=5, stride=2, padding=2, padding_mode='zeros', bias=True),
           nn.BatchNorm2d(channels[3]),
           nn.ReLU(),
           nn.Conv2d(channels[3], channels[4], kernel_size=5, stride=2, padding=2, padding_mode='zeros', bias=True),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           Flatten(),
           nn.Linear(channels[4]*img_dim*img_dim, channels[5]),
           nn.Sigmoid()
        )