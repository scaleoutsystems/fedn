import os
import traceback

import torch
from torch import nn

from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase
from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)

seed = 42
torch.manual_seed(seed)


class ServerModel(nn.Module):
    """Server side neural network model for Split Learning."""

    def __init__(self, input_features):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(input_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc(x))
        return x


class Aggregator(AggregatorBase):
    """Local SGD / Federated Averaging (FedAvg) aggregator. Computes a weighted mean
        of parameter updates.

    :param id: A reference to id of :class: `fedn.network.combiner.Combiner`
    :type id: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    :param control: A handle to the :class: `fedn.network.combiner.roundhandler.RoundHandler`
    :type control: class: `fedn.network.combiner.roundhandler.RoundHandler`

    """

    def __init__(self, update_handler):
        """Constructor method"""
        super().__init__(update_handler)

        self.name = "splitlearningagg"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def combine_models(self, helper=None, delete_models=True, is_sl_inference=False):
        """Concatenates client embeddings in the queue by aggregating them.

        After all embeddings are received, the embeddings need to be sorted
        (consistently) by client ID.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`, optional
        :param delete_models: Delete models from storage after aggregation, defaults to True
        :type delete_models: bool, optional
        :param is_sl_inference: Whether it is a splitlearning inference session (no gradient calculation) or not
        :type is_sl_inference: bool
        :return: The gradients and metadata
        :rtype: tuple
        """
        data = {}
        data["time_model_load"] = 0.0
        data["time_model_aggregation"] = 0.0

        embeddings = None
        nr_aggregated_embeddings = 0

        logger.info("AGGREGATOR({}): Aggregating client embeddings... ".format(self.name))

        while not self.update_handler.model_updates.empty():
            try:
                logger.info("AGGREGATOR({}): Getting next embedding from queue.".format(self.name))
                new_embedding = self.update_handler.next_model_update()  # returns in format {client_id: embedding}

                # Load model parameters and metadata
                logger.info("AGGREGATOR({}): Loading embedding metadata.".format(self.name))
                embedding_next, metadata = self.update_handler.load_model_update(new_embedding, helper)

                logger.info("AGGREGATOR({}): Processing embedding metadata: {}  ".format(self.name, metadata))

                if nr_aggregated_embeddings == 0:
                    embeddings = embedding_next
                else:
                    embeddings = helper.increment_average(embeddings, embedding_next)

                nr_aggregated_embeddings += 1
                # Delete model from storage
                if delete_models:
                    self.update_handler.delete_model(new_embedding)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"AGGREGATOR({self.name}): Error encoutered while processing embedding update: {e}")
                logger.error(tb)

        logger.info("splitlearning aggregator: Embeddings have been aggregated.")

        result = {"gradients": None, "validation_data": None, "data": None}

        # order embeddings and change to tensor
        client_order = sorted(embeddings.keys())
        ordered_embeddings = []
        for client_id in client_order:
            embedding = torch.tensor(embeddings[client_id], requires_grad=True)
            ordered_embeddings.append(embedding)

        concatenated_embeddings = torch.cat(ordered_embeddings, dim=1)  # to 1d tensor

        # instantiate server model
        if self.model is None:
            self.input_features = concatenated_embeddings.shape[1]
            self.model = ServerModel(self.input_features)
            self.model.to(self.device)

        # check if concatenated_embeddings matches the input features of the server model
        if concatenated_embeddings.shape[1] != self.input_features:
            logger.error(
                f"Server-side input feature mismatch: Received {concatenated_embeddings.shape[1]} input features, but expected {self.input_features}. \
                This is likely because one of the clients dropped out."
            )
            raise ValueError

        if is_sl_inference == "False":
            # split learning forward pass with gradient calculation
            logger.info("Split Learning Aggregator: Executing forward training pass")

            gradients = self.calculate_gradients(concatenated_embeddings, client_order, ordered_embeddings)

            result["gradients"] = gradients
            result["data"] = data

            logger.info("AGGREGATOR({}): Gradients are calculated.".format(self.name))

            return result
        else:
            # split learning forward pass for inference, no gradient calculation (used for validation)
            logger.info("Split Learning Aggregator: Executing forward inference pass")

            validation_data = self.calculate_validation_metrics(concatenated_embeddings)

            result["validation_data"] = validation_data
            result["data"] = data

            logger.info("AGGREGATOR({}): Test Loss: {}, Test Accuracy: {}".format(self.name, validation_data["test_loss"], validation_data["test_accuracy"]))

            return result

    def calculate_gradients(self, concatenated_embeddings, client_order, ordered_embeddings):
        self.model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        optimizer.zero_grad()

        output = self.model(concatenated_embeddings)
        targets = self.load_targets(is_train=True)
        targets = targets.to(self.device)

        loss = criterion(output, targets)
        logger.info("AGGREGATOR({}): Train Loss: {}".format(self.name, loss))

        loss.backward()

        optimizer.step()

        # Split gradients by client
        gradients = {}
        for client_id, embedding in zip(client_order, ordered_embeddings):
            gradients[str(client_id)] = embedding.grad.numpy()

        return gradients

    def calculate_validation_metrics(self, concatenated_embeddings):
        self.model.eval()
        with torch.no_grad():
            criterion = nn.BCELoss()
            output = self.model(concatenated_embeddings)
            targets = self.load_targets(is_train=False)
            targets = targets.to(self.device)
            # metric calculation
            test_loss = criterion(output, targets)

            predictions = (output > 0.5).float()
            correct = (predictions == targets).sum().item()
            total = targets.numel()  # Total number of predictions
            test_accuracy = correct / total

            validation_data = {"test_loss": test_loss, "test_accuracy": test_accuracy}

            return validation_data

    def load_targets(self, is_train=True):
        """Load target labels for split learning."""
        try:
            data_path = os.environ.get("FEDN_LABELS_PATH")
        except Exception as e:
            logger.error(f"FEDN_LABELS_PATH environment variable is not set. Set via export FEDN_LABELS_PATH='path/to/labels.pt', {e}")
            raise

        try:
            data = torch.load(data_path, weights_only=True)
            if is_train:
                targets = data["y_train"]
            else:
                targets = data["y_test"]
            return targets.reshape(-1, 1)  # Reshape to match model output shape
        except Exception as e:
            logger.error(f"Error loading labels from {data_path}: {str(e)}")
            raise
