import time
import traceback

from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase


class Aggregator(AggregatorBase):
    """Local SGD / Federated Averaging (FedAvg) aggregator. Computes a weighted mean
        of parameter updates.

    :param control: A handle to the :class: `fedn.network.combiner.updatehandler.UpdateHandler`
    :type control: class: `fedn.network.combiner.updatehandler.UpdateHandler`
    """

    def __init__(self, update_handler):
        """Constructor method"""
        super().__init__(update_handler)

        self.name = "fedavg"

    def combine_models(self, helper=None, delete_models=True, parameters=None):
        """Aggregate all model updates in the queue by computing an incremental
        weighted average of model parameters.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`, optional
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
        data["time_model_load"] = 0.0
        data["time_model_aggregation"] = 0.0

        model = None
        nr_aggregated_models = 0
        total_examples = 0

        logger.info("AGGREGATOR({}): Aggregating model updates... ".format(self.name))

        while not self.update_handler.model_updates.empty():
            try:
                logger.info("AGGREGATOR({}): Getting next model update from queue.".format(self.name))
                model_update = self.update_handler.next_model_update()

                # Load model parameters and metadata
                logger.info("AGGREGATOR({}): Loading model metadata {}.".format(self.name, model_update.model_update_id))

                tic = time.time()
                model_next, metadata = self.update_handler.load_model_update(model_update, helper)
                data["time_model_load"] += time.time() - tic

                logger.info("AGGREGATOR({}): Processing model update {}, metadata: {}  ".format(self.name, model_update.model_update_id, metadata))

                # Increment total number of examples
                total_examples += metadata["num_examples"]

                tic = time.time()
                if nr_aggregated_models == 0:
                    model = model_next
                else:
                    model = helper.increment_average(model, model_next, metadata["num_examples"], total_examples)
                data["time_model_aggregation"] += time.time() - tic

                nr_aggregated_models += 1
                # Delete model from storage
                if delete_models:
                    self.update_handler.delete_model(model_update)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"AGGREGATOR({self.name}): Error encoutered while processing model update: {e}")
                logger.error(tb)

        data["nr_aggregated_models"] = nr_aggregated_models

        logger.info("AGGREGATOR({}): Aggregation completed, aggregated {} models.".format(self.name, nr_aggregated_models))
        return model, data
