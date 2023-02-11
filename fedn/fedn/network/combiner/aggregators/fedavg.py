import queue
import time

import fedn.common.net.grpc.fedn_pb2 as fedn
from fedn.network.combiner.aggregators.aggregator import Aggregator


class FedAvg(Aggregator):
    """ Local SGD / Federated Averaging (FedAvg) aggregator. Computes a weighted mean 
        of parameter updates. 

    :param id: A reference to id of :class: `fedn.combiner.Combiner`
    :type id: str
    :param storage: Model repository for :class: `fedn.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.combiner.Combiner`
    :type server: class: `fedn.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.clients.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.clients.combiner.modelservice.ModelService`
    :param control: A handle to the :class: `fedn.clients.combiner.roundcontrol.RoundControl`
    :type control: class: `fedn.clients.combiner.roundcontrol.RoundControl`

    """

    def __init__(self, id, storage, server, modelservice, control):
        """Constructor method
        """

        super().__init__(id, storage, server, modelservice, control)

        self.name = "FedAvg"

    def combine_models(self, helper=None, time_window=180, max_nr_models=100):
        """Aggregate client model updates in the queue by computing an incremental
           weighted average of parameters.

        :param helper: An instance of :class: `fedn.utils.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.HelperBase`, optional
        :param time_window: The time window for model aggregation, defaults to 180
        :type time_window: int, optional
        :param max_nr_models: The maximum number of updates aggregated, defaults to 100
        :type max_nr_models: int, optional
        :return: The global model and metadata
        :rtype: tuple
        """

        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0

        model = None
        nr_aggregated_models = 0
        total_examples = 0

        self.server.report_status(
            "AGGREGATOR({}): Aggregating model updates... ".format(self.name))

        while not self.model_updates.empty():
            try:
                # Get next model from queue
                model_next, update_data, model_id = self.next_model_update(helper)
                self.server.report_status(
                    "AGGREGATOR({}): Processing model update {}, update_data: {}  ".format(self.name, model_id, update_data))

                # Increment total number of examples
                total_examples += update_data['num_examples']

                if nr_aggregated_models == 0:
                    model = model_next
                else:
                    model = helper.increment_average(
                        model, model_next, update_data['num_examples'], total_examples)

                nr_aggregated_models += 1
                self.model_updates.task_done()
            except Exception as e:
                self.server.report_status(
                    "AGGREGATOR({}): Error encoutered while processing model update {}, skipping this update.".format(self.name, e))
                self.model_updates.task_done()

        data['nr_aggregated_models'] = nr_aggregated_models

        self.server.report_status("AGGREGATOR({}): Aggregation completed, aggregated {} models.".format(self.name, nr_aggregated_models),
                                  log_level=fedn.Status.INFO)
        return model, data
