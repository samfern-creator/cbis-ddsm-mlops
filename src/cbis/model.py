import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(pretrained=True):
    """ResNet18 with one output (a logit for 'malignant')."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
