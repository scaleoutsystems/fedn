import os
import sys

import torch
from data import load_data
from model import load_parameters, save_parameters
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW

from fedn.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

def validate(in_model_path, out_json_path, data_path=None):
    """ Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    _, _, test_texts, test_labels = load_data(data_path)


    # encode
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1) # index of the max logit
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("correct: ", correct)

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # JSON schema
    report = {
        "test_accuracy": accuracy
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
