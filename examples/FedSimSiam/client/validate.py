import os
import sys

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from model import load_parameters
from data import load_data, load_knn_monitoring_dataset
from monitoring import *
from fedn.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


class Cifar10(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # Approx. CIFAR-10 means
                                 std=[0.247, 0.243, 0.261])  # Approx. CIFAR-10 std deviations
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.fromarray(x.astype(np.uint8))
        x = self.transform(x)
        y = self.y[idx]
        return x, y


class LinearEvaluationSimSiam(nn.Module):
    def __init__(self, in_model_path):
        super(LinearEvaluationSimSiam, self).__init__()
        model = load_parameters(in_model_path)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = model.encoder.to(device)

        # freeze parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(2048, 10).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def linear_evaluation(in_model_path, out_json_path, data_path=None, train_data_percentage=0.1, epochs=5):
    model = LinearEvaluationSimSiam(in_model_path)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    x_train, y_train = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)

    # for linear evaluation, train only on small subset of training data
    n_training_data = train_data_percentage * len(x_train)
    print("number of training points: ", n_training_data)

    x_train = x_train[:int(n_training_data)]
    y_train = y_train[:int(n_training_data)]
    print(len(x_train))

    traindata = Cifar10(x_train, y_train)
    trainloader = DataLoader(traindata, batch_size=4, shuffle=True)

    testdata = Cifar10(x_test, y_test)
    testloader = DataLoader(testdata, batch_size=4, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    model.encoder.eval()   # this is linear evaluation, only train the classifier
    model.classifier.train()

    for epoch in range(epochs):
        correct = 0
        total = 0
        total_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                features = model.encoder(inputs)
            outputs = model.classifier(features)

            loss = criterion(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)

    training_accuracy = correct / total
    print(f"Accuracy: {training_accuracy * 100:.2f}%")

    training_loss = total_loss / total
    print("train loss: ", training_loss)

    # test on test_set
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

    test_accuracy = correct_preds / total_samples
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    test_loss = total_loss / total_samples
    print("test loss: ", test_loss)

    return training_loss, training_accuracy, test_loss, test_accuracy


def validate(in_model_path, out_json_path, data_path=None, train_data_percentage=1, epochs=3):

    memory_loader, test_loader = load_knn_monitoring_dataset(data_path)

    model = load_parameters(in_model_path)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    knn_accuracy = knn_monitor(model.encoder, memory_loader, test_loader, device, k=min(
        25, len(memory_loader.dataset)))

    print("knn accuracy: ", knn_accuracy)

    # JSON schema
    report = {
        "knn_accuracy": knn_accuracy,
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
