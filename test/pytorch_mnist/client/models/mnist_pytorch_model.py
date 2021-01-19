from torch import nn
import torch


import tempfile

# Create an initial CNN Model
def create_seed_model():
    model = CNN()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1, rho=0.95, eps=1e-07)
    return model, loss, optimizer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32,
                              kernel_size=3, stride=1, padding=0)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=3, stride=1, padding=0)
        self.act2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.flatten = lambda x: x.view(x.size(0), -1)

        self.dense1 = nn.Linear(64*12*12, 128)
        self.act3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(128, 10)
        self.output_act = nn.Softmax(dim=0)


    def forward(self, x):
        x = self.cnn1(x)
        x = self.act1(x)

        x = self.cnn2(x)
        x = self.act2(x)

        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.act3(x)
        x = self.dropout2(x)

        x = self.dense2(x)
        x = self.output_act(x)
        return x
