from torchvision.models import resnet50
import torch.nn as nn

def lcn_model():
    model = resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    )
    return model
