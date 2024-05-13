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


def validate(in_model_path, out_json_path, data_path=None):

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
