import numpy as np
import torch
from torch.utils import data

import mia
from mia import train
from mia import dataset
from mia.dataset import challenge_set

N_SHADOW_MODELS = 16
EPOCHS = 20

if __name__ != "__main__":
    exit(0)

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for shadow_idx in range(N_SHADOW_MODELS):
    print(f"training shadow model {shadow_idx}")
    # subset the challenge dataset randomly
    # randomly partition challenge set into train/test
    indicators = np.random.choice([0, 1], size=len(challenge_set))
    
    trainset_idx = np.argwhere(indicators).squeeze()
    testset_idx = np.argwhere(1-indicators).squeeze()

    shadow_trainset = data.Subset(challenge_set, trainset_idx)
    shadow_testset = data.Subset(challenge_set, testset_idx)

    # attr dict
    attr = {"shadow_idx": shadow_idx, "indicators": indicators}

    shadow_model = mia.create_model()
    train.train_model(shadow_model, 
                      f"shadow{shadow_idx}",
                      dataset.create_trainloader(shadow_trainset),
                      dataset.create_testloader(shadow_testset),
                      n_epochs = EPOCHS,
                      save_ckpt=True,
                      save_path="./shadowmodels/",
                      attr=attr
                      )