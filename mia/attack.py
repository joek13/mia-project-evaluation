import numpy as np
import torch

def loss_attack(model, record, criterion, threshold):
    # simple LOSS attack due to Yeom et al.
    # as described in pp.3 of Carlini et al.
    model.eval()
    with torch.no_grad():
        image, label = record
        outputs = model(image[None, :])
        loss = criterion(outputs.squeeze(), torch.tensor(label))

    return loss < threshold

def compute_loss_threshold(model, criterion, trainloader, percentile, n=10_000):
    # samples n training instances and returns p^th percentile of loss
    with torch.no_grad():
        losses = []

        for (images, labels) in trainloader:
            outputs = model(images)
            for (output, label) in zip(outputs, labels):
                losses.append(criterion(output, label))

            if len(losses) > n:
                break

        return np.percentile(losses, percentile)