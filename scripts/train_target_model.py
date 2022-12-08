# trains standard resnet model on cifar-10 dataset

import numpy as np

import torch
import torch.optim as optim

from torchvision.models import resnet
from torchvision import datasets, transforms

BATCH_SIZE = 4
EPOCHS = 2

if __name__ != "__main__":
    exit(0)

# resnet 18 model
model = resnet.resnet18(num_classes=10)

# dataset transforms
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# use CIFAR10 training/test set
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def print_statistics(epoch, running_loss, over_minibatches):
    test_loss = 0.0
    test_correct = 0
    n_test = 0

    with torch.no_grad():
        for (inputs, labels) in testloader:
            outputs = model(inputs)
            test_correct += np.count_nonzero(np.argmax(outputs, axis=1) == labels)
            test_loss += criterion(outputs, labels).item()
            n_test += len(inputs)

    print(f"[{epoch + 1}] train loss: {running_loss / over_minibatches:.3f} test loss: {test_loss / n_test:.3f} test accuracy: {test_correct / n_test:.3f}")

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print_statistics(epoch, running_loss, 2000)
            running_loss = 0.0
    
    PATH = f'./models/resnet_{epoch + 1}.pth'
    torch.save(model.state_dict(), PATH)
        