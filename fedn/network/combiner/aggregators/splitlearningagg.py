import traceback

import torch
from torch import nn

from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase
from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)


class ServerModel(nn.Module):
    """Server side neural network model for Split Learning."""

    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(12, 6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


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
        self.model = ServerModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCEWithLogitsLoss()

    def combine_models(self, helper=None, delete_models=True):
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
                new_embedding = self.update_handler.next_model_update() # NOTE: should return in format {client_id: embedding}

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

        # NOTE: When aggregating the embeddings in SplitLearning, they always need to be sorted consistently
        client_order = sorted(embeddings.keys())

        # to tensor
        ordered_embeddings = [torch.from_numpy(embeddings[k]).float().requires_grad_(True) for k in client_order] # list of embeddings, without client_id

        # Continue forward pass
        concatenated_embeddings = torch.cat(ordered_embeddings, dim=1) # to 1d tensor

        self.optimizer.zero_grad()

        output = self.model(concatenated_embeddings)

        # TODO: need to match indices of data samples to target indices in order to calculate gradients.
        # NOTE: For one epoch, depending on the batch size, multiple communications are necessary.
        # use dummy target for now
        # batch_size = concatenated_embeddings.shape[0] # TODO: check

        targets = helper.load_targets()

        loss = self.criterion(output, targets)
        loss.backward()

        self.optimizer.step()

        logger.info("AGGREGATOR({}): Loss: {}".format(self.name, loss))

        # Split gradients according to original client order
        gradients = {}
        for client_id, embedding in zip(client_order, ordered_embeddings):
            gradients[str(client_id)] = embedding.grad.numpy()

        logger.info("AGGREGATOR({}): Gradients are calculated.".format(self.name))
        return gradients, data
