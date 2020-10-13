import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def read_data(filename, sample_fraction=1.0):
    data = np.array(pd.read_csv(filename, header=None))
    X = data[:, 1::]
    y = data[:, 0]

    # Subsample data if required
    if sample_fraction < 1.0:
        _, X, _, y = train_test_split(X, y, test_size=sample_fraction)

    # Transform data into torch tensor
    X = X.reshape((X.shape[0], 1, 28, 28))
    X = X.astype('float32')
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return  X, y

def create_seed_model():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.mp = nn.MaxPool2d(2)

            self.fc = nn.Linear(320, 10)

        def forward(self, x):
            in_size = x.size(0)
            x = F.relu(self.mp(self.conv1(x)))
            x = F.relu(self.mp(self.conv2(x)))
            x = x.view(in_size, -1)
            x = self.fc(x)
            return F.log_softmax(x)

    model = Net()
    optimizer = optim.Adadelta(model.parameters())
    loss = F.nll_loss
    return model, optimizer, loss

def read_model(path):
    model, optimizer, loss = create_seed_model()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model, optimizer, loss