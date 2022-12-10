import numpy as np
import tqdm

import torch
from torch import optim
from pathlib import Path

from mia import save_model

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_statistics(model, test_loader, epoch, train_loss, criterion):
    test_loss = 0.0
    test_correct = 0
    n_test = 0

    with torch.no_grad():
        for (inputs, labels) in tqdm.tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            test_correct += np.count_nonzero(np.argmax(outputs.cpu(), axis=1) == labels.cpu())
            test_loss += criterion(outputs, labels).item()
            n_test += len(inputs)

    print(f"[epoch {epoch + 1}] train loss: {train_loss:.3f} test loss: {test_loss / n_test:.3f} test accuracy: {test_correct / n_test:.3f}")
    return test_correct / n_test



def train_model(model, moniker, train_loader, test_loader=None, epoch_start=0, n_epochs=50, lr=0.1, save_ckpt=True, save_path="./models/", attr={}):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(epoch_start, epoch_start + n_epochs):
        running_loss = 0.0
        instances = 0
        acc = 0
        
        for i, data in tqdm.tqdm(enumerate(train_loader), f"epoch {epoch+1}"):
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

        scheduler.step()

        if test_loader is not None:
            acc = print_statistics(model, test_loader, epoch, running_loss / instances, criterion)

        if save_ckpt:
            ckpt_dir = Path(save_path).joinpath("ckpt")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir.joinpath(f"{moniker}_epoch{epoch + 1}.pth")
            save_model(model, ckpt_path, acc, attr)
            print(f"saved checkpoint to {ckpt_path}")
    
    model_dir = Path(save_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir.joinpath(f"{moniker}.pth")
    save_model(model, model_path, acc, attr)
    print(f"saved model to {model_path}")