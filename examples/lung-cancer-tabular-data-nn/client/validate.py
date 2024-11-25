import os
import sys

import numpy as np
from data import load_data
from model import load_parameters
from fedn.utils.helpers.helpers import save_metrics
import collections


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Function to calculate accuracy

def calculate_accuracy(outputs, labels):
    
    predicted = (outputs >= 0.5).int()
    labels = labels.int()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy * 100


def validate(in_model_path, out_json_path, data_path=None):
    
    # Load data
    X_train, y_train = load_data(data_path)
    X_test, y_test = load_data(data_path, is_train=False)

    train_dataset = TensorDataset(X_train , y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

    # Load model
    model = load_parameters(in_model_path)


    # Evaluate
 
    model.eval()
   
    # Calculate training loss and accuracy
    train_loss = 0.0
    train_accuracy = 0.0
    total_train_samples = 0

    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            train_loss += loss.item() * labels.size(0)
            train_accuracy += calculate_accuracy(outputs, labels) * labels.size(0)
            total_train_samples += labels.size(0)

    # Average training metrics
    avg_train_loss = train_loss / total_train_samples
    avg_train_accuracy = train_accuracy / total_train_samples

    # Calculate test loss and accuracy
    test_loss = 0.0
    test_accuracy = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)
            test_accuracy += calculate_accuracy(outputs, labels) * labels.size(0)
            total_test_samples += labels.size(0)

    # Average test metrics
    avg_test_loss = test_loss / total_test_samples
    avg_test_accuracy = test_accuracy / total_test_samples

    # Print metrics
    print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.2f}%')
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.2f}%')
    print('-' * 50)

    # JSON schema
    report = {
        "training_loss": avg_train_loss,
        "training_accuracy": avg_train_accuracy,
        "test_loss": avg_test_loss,
        "test_accuracy": avg_test_accuracy,
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
