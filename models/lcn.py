from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def lcn_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    return model