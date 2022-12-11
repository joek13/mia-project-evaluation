# trains standard resnet model on cifar-10 dataset
import numpy as np
import tqdm

import torch
import torch.optim as optim

from torchvision import datasets, transforms

import mia
from mia.dataset import trainset, testset, trainloader, testloader
from mia import train

EPOCHS = 20

if __name__ != "__main__":
    exit(0)

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = mia.create_model().to(device)

train.train_model(model, 
                  "resnet18",
                  trainloader,
                  testloader,
                  n_epochs=EPOCHS)