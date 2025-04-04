import io

from data import get_dataloaders
from model import compile_model, load_parameters, model_hyperparams, save_parameters
from settings import BATCHSIZE_VALID, DATASET_PATH, DATASET_TOTAL_SPLITS, KEYWORDS
from torch import nn
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy

from fedn.network.clients.fedn_client import FednClient


class SCClient(FednClient):
    def __init__(self, dataset_split_idx):
        super().__init__(self.train, self.validate, self.predict)

        self.dataset_split_idx = dataset_split_idx

    def train(self, model_params, settings):
        """Train the model with the given parameters and settings.

        Args:
            model_params: The model parameters.
            settings: The training settings.
            dataset_split_idx: The index of the dataset split to use.

        Returns:
            tuple: The trained model parameters and metadata.

        """
        training_metadata = {"batchsize_train": 64, "lr": 1e-3, "n_epochs": 1}

        dataloader_train, dataloader_valid, _ = get_dataloaders(
            DATASET_PATH, KEYWORDS, self.dataset_split_idx, DATASET_TOTAL_SPLITS, training_metadata["batchsize_train"], BATCHSIZE_VALID
        )

        sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
        load_parameters(sc_model, model_params)

        optimizer = Adam(sc_model.parameters(), lr=training_metadata["lr"])
        loss_fn = nn.CrossEntropyLoss()
        n_epochs = training_metadata["n_epochs"]

        for epoch in range(n_epochs):
            sc_model.train()
            for idx, (y_labels, x_spectrograms) in enumerate(dataloader_train):
                optimizer.zero_grad()
                _, logits = sc_model(x_spectrograms)

                loss = loss_fn(logits, y_labels)
                loss.backward()

                optimizer.step()

                if idx % 100 == 0:
                    acc = self.evaluate(dataloader_valid, sc_model)
                    print(f"Epoch {epoch + 1}/{n_epochs} | Batch: {idx + 1}/{len(dataloader_train)} | Loss: {loss.item()} | Training Acc: {acc:.4f}")
                    self.log_metric({"loss": loss.item(), "Training acc": acc})

        out_model = save_parameters(sc_model, io.BytesIO())

        metadata = {"training_metadata": {"num_examples": len(dataloader_train.dataset)}}

        return out_model, metadata

    def validate(self, model_params) -> dict:
        """Validate the model with the given parameters and dataset split index.

        Args:
            model_params: The model parameters.
            dataset_split_idx: The index of the dataset split to use.

        Returns:
            dict: The validation report containing training and validation accuracy.

        """
        dataloader_train, dataloader_valid, _ = get_dataloaders(
            DATASET_PATH, KEYWORDS, self.dataset_split_idx, DATASET_TOTAL_SPLITS, BATCHSIZE_VALID, BATCHSIZE_VALID
        )

        sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
        load_parameters(sc_model, model_params)

        return {"training_acc": self.evaluate(dataloader_train, sc_model), "validation_acc": self.evaluate(dataloader_valid, sc_model)}

    def evaluate(self, dataloader, model) -> float:
        n_labels = dataloader.dataset.n_labels
        model.eval()
        metric = MulticlassAccuracy(num_classes=n_labels)
        for y_labels, x_spectrograms in dataloader:
            probs, _ = model(x_spectrograms)

            y_pred = probs.argmax(-1)
            metric.update(y_pred, y_labels)
        return metric.compute().item()

    def predict(self, model_params) -> dict:
        return {}
