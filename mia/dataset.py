import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, datasets

BATCH_SIZE = 32

# dataset transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transform_test
])

def create_trainloader(set):
    return torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE,
                                       shuffle=True, num_workers=0)

def create_testloader(set):
    return torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=0)

# use CIFAR10 training/test set
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = create_trainloader(trainset)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = create_testloader(testset)

# challenge set for MIA's to sort into train/test
# 10k images from train and test set, respectively
# for simplicity, we use a fixed challenge set
_subset_train = data.Subset(trainset, np.arange(10_000))
_subset_test = data.Subset(testset, np.arange(10_000))
challenge_set = data.ConcatDataset([_subset_train, _subset_test])