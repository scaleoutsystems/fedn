import math
import os
import sys

import torch
from data import load_data
from model import load_parameters, save_parameters
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW

from fedn.utils.helpers.helpers import save_metadata


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
    

def train(in_model_path, out_model_path, data_path=None, batch_size=16, epochs=1, lr=5e-5):
    """ Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
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
    train_texts, train_labels, _, _ = load_data(data_path)

    # encode
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = IMDbDataset(train_encodings, train_labels)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optim = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            print(loss)
            loss.backward()
            optim.step()


    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        'num_examples': len(train_dataset),
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
    # train("../seed.npz", "../seed.npz")
