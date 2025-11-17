import os
import sys

import torch
import yaml
from data import DATA_CLASSES, MedNISTDataset
from model import load_parameters
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    ScaleIntensity,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score

from scaleout.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])


def validate(in_model_path, out_json_path, data_path=None, client_settings_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param client_settings_path: The path to the local client settings file.
    :type client_settings_path: str
    """
    if client_settings_path is None:
        client_settings_path = os.environ.get("SCALEOUT_CLIENT_SETTINGS_PATH", dir_path + "/client_settings.yaml")

    with open(client_settings_path, "r") as fh:  # Used by CJG for local training
        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError:
            raise

    num_workers = client_settings["num_workers"]
    batch_size = client_settings["batch_size"]
    split_index = os.environ.get("SCALEOUT_DATA_SPLIT_INDEX")

    if data_path is None:
        data_path = os.environ.get("SCALEOUT_DATA_PATH")

    with open(os.path.join(os.path.dirname(data_path), "data_splits.yaml"), "r") as file:
        clients = yaml.safe_load(file)

    image_list = clients["client " + str(split_index)]["validation"]

    val_ds = MedNISTDataset(data_path=data_path + "/MedNIST/", transforms=val_transforms, image_files=image_list)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = (
                val_data[0].to(device),
                val_data[1].to(device),
            )
            pred = model(val_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(val_labels[i].item())
                y_pred.append(pred[i].item())

    class_names = list(DATA_CLASSES.keys())
    print("class names: ", class_names)
    cr = classification_report(y_true, y_pred, digits=4, output_dict=True, target_names=class_names)
    report = {class_name + "_" + metric: cr[class_name][metric] for class_name in cr if isinstance(cr[class_name], dict) for metric in cr[class_name]}
    report.update({class_name: cr[class_name] for class_name in cr if isinstance(cr[class_name], str)})

    # JSON schema
    report.update({"test_accuracy": accuracy_score(y_true, y_pred), "test_f1_score": f1_score(y_true, y_pred, average="macro")})

    for key, value in report.items():
        print(f"{key}: {value}")

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
