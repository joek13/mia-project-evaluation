import torch
from torchvision.models import resnet

def create_model():
    # resnet18 with 10 classes
    return resnet.resnet18(num_classes=10)

def load_model(path, device=None):
    model = create_model()
    model.load_state_dict(torch.load(path, map_location=device))
    return model