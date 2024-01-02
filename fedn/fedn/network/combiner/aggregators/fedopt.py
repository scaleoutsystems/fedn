import math

import numpy as np

from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase


class Aggregator(AggregatorBase):
    """ Federated Optimization (FedOpt) aggregator.

    Implmentation following: https://arxiv.org/pdf/2003.00295.pdf

    Aggregate pseudo gradients computed by subtracting the model update
    from the global model weights from the previous round.

    :param id: A reference to id of :class: `fedn.network.combiner.Combiner`
    :type id: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    :param control: A handle to the :class: `fedn.network.combiner.round.RoundController`
    :type control: class: `fedn.network.combiner.round.RoundController`

    """

    def __init__(self, storage, server, modelservice, control):
        """Constructor method"""

        super().__init__(storage, server, modelservice, control)

        self.name = "fedopt"
        # Server side hyperparameters
        self.eta = 1
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.tau = 1e-3

    def combine_models(self, helper=None, time_window=180, max_nr_models=100, delete_models=True):
        """Compute pseudo gradients usigng model updates in the queue.

        :param helper: An instance of :class: `fedn.utils.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.HelperBase`, optional
        :param time_window: The time window for model aggregation, defaults to 180
        :type time_window: int, optional
        :param max_nr_models: The maximum number of updates aggregated, defaults to 100
        :type max_nr_models: int, optional
        :param delete_models: Delete models from storage after aggregation, defaults to True
        :type delete_models: bool, optional
        :return: The global model and metadata
        :rtype: tuple
        """

        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0

        model = None
        nr_aggregated_models = 0
        total_examples = 0

        logger.info(
            "AGGREGATOR({}): Aggregating model updates... ".format(self.name))

        # v = math.pow(self.tau, 2)
        # m = 0.0

        while not self.model_updates.empty():
            try:
                # Get next model from queue
                model_next, metadata, model_id, model_update = self.next_model_update(helper)
                logger.info(
                    "AGGREGATOR({}): Processing model update {}, metadata: {}  ".format(self.name, model_id, metadata))
                print("***** ", model_update, flush=True)

                # Increment total number of examples
                total_examples += metadata['num_examples']

                if nr_aggregated_models == 0:
                    model_old = self.control.load_model_update(helper, model_update.model_id)
                    pseudo_gradient = helper.subtract(model_next, model_old)
                else:
                    pseudo_gradient_next = helper.subtract(model_next, model_old)
                    pseudo_gradient = helper.increment_average(
                        pseudo_gradient, pseudo_gradient_next, metadata['num_examples'], total_examples)

                print("NORM PSEUDOGRADIENT: ", helper.norm(pseudo_gradient), flush=True)

                nr_aggregated_models += 1
                # Delete model from storage
                if delete_models:
                    self.modelservice.models.delete(model_id)
                    logger.info(
                        "AGGREGATOR({}): Deleted model update {} from storage.".format(self.name, model_id))
                self.model_updates.task_done()
            except Exception as e:
                logger.error(
                    "AGGREGATOR({}): Error encoutered while processing model update {}, skipping this update.".format(self.name, e))
                self.model_updates.task_done()

        # Server-side momentum
        # m = helper.add(m, pseudo_gradient, self.beta1, (1.0-self.beta1))
        # v = v + helper.power(pseudo_gradient, 2)
        # model = model_old + self.eta*m/helper.sqrt(v+self.tau)

        model = helper.add(model_old, pseudo_gradient, 1.0, self.eta)

        data['nr_aggregated_models'] = nr_aggregated_models

        logger.info("AGGREGATOR({}): Aggregation completed, aggregated {} models.".format(self.name, nr_aggregated_models))
        return model, data
