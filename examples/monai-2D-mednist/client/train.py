import math
import os
import sys

import torch
from model import load_parameters, save_parameters
from monai.metrics import ROCAUCMetric
from data import load_data, get_classes
from fedn.utils.helpers.helpers import save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

from monai.data import decollate_batch, DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
import numpy as np
from monai.data import decollate_batch, DataLoader

def pre_training_settings(num_class, batch_size, train_x, train_y, val_x, val_y):

    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )

    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    class MedNISTDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]


    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    return train_ds, train_loader, val_ds, val_loader, y_pred_trans, y_trans


def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data directory.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """
    max_epochs = epochs
    val_interval = 1
    auc_metric = ROCAUCMetric()

    num_class = len(get_classes(data_path))
    # Load data
    x_train, y_train, x_val, y_val = load_data(data_path)
    train_ds, train_loader, val_ds, val_loader, y_pred_trans, y_trans = pre_training_settings(num_class, batch_size, x_train, y_train, x_val, y_val)
    

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    loss_function = torch.nn.CrossEntropyLoss()
    # Train
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    # writer = SummaryWriter()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join("best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                #writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    #writer.close()


    #print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
