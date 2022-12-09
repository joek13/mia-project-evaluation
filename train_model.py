# trains standard resnet model on cifar-10 dataset
import numpy as np
import tqdm

import torch
import torch.optim as optim

from torchvision import datasets, transforms

import mia

BATCH_SIZE = 32
EPOCHS = 50

START_FROM = 0

if __name__ != "__main__":
    exit(0)

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if START_FROM == 0:
    model = mia.create_model().to(device)
else:
    model = mia.load_model(f"./models/resnet_epoch{START_FROM}.pth").to(device)

# dataset transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# use CIFAR10 training/test set
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def print_statistics(epoch, train_loss):
    test_loss = 0.0
    test_correct = 0
    n_test = 0

    with torch.no_grad():
        for (inputs, labels) in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            test_correct += np.count_nonzero(np.argmax(outputs.cpu(), axis=1) == labels.cpu())
            test_loss += criterion(outputs, labels).item()
            n_test += len(inputs)

    print(f"[epoch {epoch + 1}] train loss: {train_loss:.3f} test loss: {test_loss / n_test:.3f} test accuracy: {test_correct / n_test:.3f}")

for epoch in range(START_FROM, START_FROM + EPOCHS):
    running_loss = 0.0
    instances = 0
    
    for i, data in tqdm.tqdm(enumerate(trainloader), f"epoch {epoch+1}"):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        instances += len(inputs)

    print_statistics(epoch, running_loss / instances)
    
    PATH = f'./models/resnet_epoch{epoch + 1}.pth'
    torch.save(model.state_dict(), PATH)
    print(f"saved checkpoint to {PATH}")
        