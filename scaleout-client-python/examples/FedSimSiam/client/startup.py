from scaleout import EdgeClient, ScaleoutModel
from scaleoututil.helpers.helpers import get_helper

from model import load_parameters, save_parameters
import torch
from torch.utils.data import DataLoader, Dataset
from data import load_data, load_knn_monitoring_dataset, prepare_data
from monitoring import knn_monitor
from utils import init_lrscheduler
import numpy as np
from PIL import Image
from torchvision import transforms

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


class SimSiamDataset(Dataset):
    def __init__(self, x, y, is_train=True):
        self.x = x
        self.y = y
        self.is_train = is_train

    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.fromarray(x.astype(np.uint8))

        y = self.y[idx]

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.247, 0.243, 0.261])
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        if self.is_train:
            transform = transforms.Compose(augmentation)

            x1 = transform(x)
            x2 = transform(x)
            return [x1, x2], y

        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])

            x = transform(x)
            return x, y

    def __len__(self):
        return len(self.x)


def startup(client: EdgeClient):
    """Initialize the client.
    
    :param client: The EdgeClient object.
    :type client: EdgeClient
    """
    prepare_data()
    MyClient(client)


class MyClient():
    def __init__(self, client: EdgeClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
    

    def train(self, scaleout_model: ScaleoutModel, settings, data_path=None, batch_size=32, epochs=1, lr=0.01):
        """Complete a model update.

        :param scaleout_model: The ScaleoutModel object containing the model.
        :type scaleout_model: ScaleoutModel
        :param data_path: The path to the data file.
        :type data_path: str
        :param batch_size: The batch size to use.
        :type batch_size: int
        :param epochs: The number of epochs to train.
        :type epochs: int
        :param lr: The learning rate to use.
        :type lr: float
        """
        # Load data
        x_train, y_train = load_data(data_path)

        # Load parameters and initialize model
        model = load_parameters(scaleout_model)

        trainset = SimSiamDataset(x_train, y_train, is_train=True)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True)

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        model.train()

        optimizer, lr_scheduler = init_lrscheduler(
            model, 500, trainloader)

        global_step = 0
        for epoch in range(epochs):
            for idx, data in enumerate(trainloader):
                images = data[0]
                optimizer.zero_grad()
                data_dict = model.forward(images[0].to(
                    device, non_blocking=True), images[1].to(device, non_blocking=True))
                loss = data_dict["loss"].mean()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                global_step += 1
                stats = {"loss": loss.item()}
                if global_step % 100 == 0:
                    self.client.log_metric(stats, step=global_step)

        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            "num_examples": len(x_train),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr
        }

        self.client.check_task_abort()

        # Save model update
        result_model = save_parameters(model)

        return result_model, {"training_metadata": metadata}


    def validate(self, scaleout_model: ScaleoutModel, data_path=None):
        """Validate model using KNN monitoring.

        :param scaleout_model: The ScaleoutModel object containing the model.
        :type scaleout_model: ScaleoutModel
        :param data_path: The path to the data file.
        :type data_path: str
        """
        memory_loader, test_loader = load_knn_monitoring_dataset(data_path)

        model = load_parameters(scaleout_model)
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        knn_accuracy = knn_monitor(model.encoder, memory_loader, test_loader, device, k=min(
            25, len(memory_loader.dataset)))

        print("knn accuracy: ", knn_accuracy)

        # Return metrics
        metrics = {
            "knn_accuracy": knn_accuracy,
        }

        return metrics