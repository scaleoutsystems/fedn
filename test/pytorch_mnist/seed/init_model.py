import os

from scaleout.project import Project

import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Create an initial CNN Model
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


if __name__ == '__main__':

# Create a seed models and push to Minio
	model, optimizer, _ = create_seed_model()
	torch.save({'model_state_dict': model.state_dict()}, "879fa112-c861-4cb1-a25d-775153e5b548")
