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
from sklearn.metrics import classification_report, accuracy_score, log_loss


# Function to calculate accuracy

def validate(in_model_path, out_json_path, data_path=None):
    
    # Load data
    X_train, y_train = load_data(data_path)
    X_test, y_test = load_data(data_path, is_train=False)


    # Load model
    model = load_parameters(in_model_path)
    

    # Evaluate

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)

    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba) 

    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # JSON schema
    report = {
        "training_loss": train_loss,
        "training_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
