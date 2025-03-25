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
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        self.test_losses = []
        self.test_accuracies = []

    def combine_models(self, helper=None, delete_models=True, is_validate=False):
        """Concatenates client embeddings in the queue by aggregating them.

        After all embeddings are received, the embeddings need to be sorted
        (consistently) by client ID.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`, optional
        :param delete_models: Delete models from storage after aggregation, defaults to True
        :type delete_models: bool, optional
        :return: The gradients and metadata
        :rtype: tuple
        """
        data = {}
        data["time_model_load"] = 0.0
        data["time_model_aggregation"] = 0.0

        embeddings = None
        nr_aggregated_embeddings = 0
        total_examples = 0

        logger.info("AGGREGATOR({}): Aggregating client embeddings... ".format(self.name))

        while not self.update_handler.model_updates.empty():
            try:
                logger.info("AGGREGATOR({}): Getting next embedding from queue.".format(self.name))
                new_embedding = self.update_handler.next_model_update()  # returns in format {client_id: embedding}

                # Load model parameters and metadata
                logger.info("AGGREGATOR({}): Loading embedding metadata.".format(self.name))
                embedding_next, metadata = self.update_handler.load_model_update(new_embedding, helper)

                logger.info("AGGREGATOR({}): Processing embedding metadata: {}  ".format(self.name, metadata))

                # Increment total number of examples
                total_examples += metadata["num_examples"]

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

        logger.info("splitlearning aggregator: starting calculation of gradients")

        # order embeddings and change to tensor
        client_order = sorted(embeddings.keys())
        ordered_embeddings = []
        for client_id in client_order:
            embedding = torch.tensor(embeddings[client_id], requires_grad=True)
            ordered_embeddings.append(embedding)

        concatenated_embeddings = torch.cat(ordered_embeddings, dim=1)  # to 1d tensor

        # instantiate server model
        if self.model is None:
            input_features = concatenated_embeddings.shape[1]
            self.model = ServerModel(input_features)
            self.model.to(self.device)

        logger.info("is_validate is {}".format(is_validate))
        logger.info("type of is_validate is {}".format(type(is_validate)))

        if is_validate == "False":  # forward training pass
            logger.info("Split Learning Aggregator: Executing forward training pass")
            # gradient calculation
            self.model.train()
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            optimizer.zero_grad()

            output = self.model(concatenated_embeddings)
            targets = helper.load_targets(is_train=True)

            logger.info("target shape: {}".format(targets.shape))
            logger.info("output shape: {}".format(output.shape))

            loss = criterion(output, targets)
            logger.info("AGGREGATOR({}): Loss: {}".format(self.name, loss))

            loss.backward()

            optimizer.step()

            # Split gradients by client
            gradients = {}
            for client_id, embedding in zip(client_order, ordered_embeddings):
                gradients[str(client_id)] = embedding.grad.numpy()

            logger.info("AGGREGATOR({}): Gradients are calculated.".format(self.name))

        else:  # split learning validation pass, we don't calculate gradients, only calculate & return test loss
            logger.info("Split Learning Aggregator: Executing forward validation pass")
            self.model.eval()
            with torch.no_grad():
                criterion = nn.BCELoss()
                output = self.model(concatenated_embeddings)
                targets = helper.load_targets(is_train=False)

                # TODO: potentially save metrics in validation store

                # test loss
                test_loss = criterion(output, targets)

                # test accuracy
                predictions = (output > 0.5).float()
                correct = (predictions == targets).sum().item()
                total = targets.numel()  # Total number of predictions
                test_accuracy = correct / total

                logger.info("AGGREGATOR({}): Test Loss: {}, Test Accuracy: {}".format(self.name, test_loss, test_accuracy))

                validation_data = {"test_loss": test_loss, "test_accuracy": test_accuracy}

                self.test_losses.append(test_loss.item())
                self.test_accuracies.append(test_accuracy)
                logger.info("AGGREGATOR({}): Test Losses: {}, Test Accuracies: {}".format(self.name, self.test_losses, self.test_accuracies))

                return validation_data, data

        return gradients, data
