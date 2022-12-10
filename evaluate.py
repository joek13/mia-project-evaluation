# evaluates trained resnet34 model on cifar10 test set

import mia
import sys
import tqdm

import numpy as np

import torch
import torch.optim as optim

from torchvision import datasets, transforms


if __name__ != "__main__":
    exit(0)

if len(sys.argv) < 2:
    print("usage: ./python evaluate.py model.pth")
    exit(1)

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = mia.load_model(sys.argv[1], device=device).to(device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)

test_loss = 0.0
test_correct = 0
n_test = 0

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for (inputs, labels) in tqdm.tqdm(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        test_correct += np.count_nonzero(np.argmax(outputs.cpu(), axis=1) == labels.cpu())
        test_loss += criterion(outputs, labels).item()
        n_test += len(inputs)

print(f"test loss: {test_loss / n_test:.3f}\ntest accuracy: {test_correct / n_test:.3f}")