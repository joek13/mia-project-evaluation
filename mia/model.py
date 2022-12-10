import torch
from torchvision.models import resnet

def create_model():
    # resnet18 with 10 classes
    return resnet.resnet18(num_classes=10)

def save_model(model, path, acc, attr):
    out_dict = {
        "state": model.state_dict(),
        "acc": acc,
        "attr": attr
    }
    torch.save(out_dict, path)

def load_model(path, device=None):
    model = create_model()

    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        rich_dict = torch.load(path, map_location=device)
        model.load_state_dict(rich_dict["state"])

    return model

def load_model_metadata(path, device=None):
    rich_dict = torch.load(path, map_location=device)
    return { "acc": rich_dict["acc"], "attr": rich_dict["attr"] }