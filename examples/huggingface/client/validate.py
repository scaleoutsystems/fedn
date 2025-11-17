import os
import sys

import torch
from data import load_data
from model import load_parameters
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scaleout.utils.helpers.helpers import save_metrics

MODEL = "google/bert_uncased_L-2_H-128_A-2"

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess(text):
    text = text.lower()
    text = text.replace("\n", " ")
    return text


def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    X_train, y_train = load_data(data_path, is_train=True)
    X_test, y_test = load_data(data_path, is_train=False)

    # preprocess
    X_test = [preprocess(text) for text in X_test]
    X_train = [preprocess(text) for text in X_train]

    # test dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    test_encodings = tokenizer(X_test, truncation=True, padding="max_length", max_length=512)
    test_dataset = SpamDataset(test_encodings, y_test)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    criterion = torch.nn.CrossEntropyLoss()

    # test set validation
    with torch.no_grad():
        correct = 0
        total_loss = 0
        total = 0
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)  # index of the max logit

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)

    test_accuracy = correct / total
    print(f"Accuracy: {test_accuracy * 100:.2f}%")

    test_loss = total_loss / total
    print("test loss: ", test_loss)

    # JSON schema
    report = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
