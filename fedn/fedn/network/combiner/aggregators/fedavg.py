import json
import queue
import time

import fedn.common.net.grpc.fedn_pb2 as fedn
from fedn.network.combiner.aggregators.aggregator import Aggregator


class FedAvg(Aggregator):
    """ Local SGD / Federated Averaging (FedAvg) aggregator.

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
        """Aggregate client model updates into combiner level model.

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

        self.server.report_status(
            "AGGREGATOR({}): Aggregating model updates...".format(self.name))

        model = None
        round_time = 0.0
        polling_interval = 1.0
        nr_aggregated_models = 0
        total_examples = 0

        # Wait until round times out, or the maximal number of models are recieved
        while round_time < time_window:
            if self.model_updates.qsize() > max_nr_models:
                break
            time.sleep(polling_interval)
            round_time += polling_interval

        while not self.model_updates.empty():
            try:
                # Get next model from queue
                model_next, update_data, model_id = self.next_model_update(helper)

                print(model_id, update_data, flush=True)

                total_examples += update_data['num_examples']

                if nr_aggregated_models == 0:
                    model = model_next
                else:
                    model = helper.increment_average(
                        model, model_next, nr_aggregated_models + 1)

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

    def combine_models_incremental(self, helper=None, time_window=180, max_nr_models=100):
        """Compute a running average of model updates.

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

        self.server.report_status(
            "AGGREGATOR({}): Aggregating model updates...".format(self.name))

        model = None
        round_time = 0.0
        polling_interval = 1.0
        nr_processed_models = 0
        while nr_processed_models < max_nr_models:
            try:
                # Get next model_id from queue
                model_update = self.model_updates.get(block=False)
                model_id = model_update.model_update_id

                self.server.report_status(
                    "AGGREGATOR({}): Received model update with id {}".format(self.name, model_id))

                # Load the model update
                tic = time.time()
                model_str = self.control.load_model_str(model_id)
                if model_str:
                    try:
                        model_next = helper.load_model_from_BytesIO(
                            model_str.getbuffer())
                    except IOError:
                        self.server.report_status(
                            "AGGREGATOR({}): Failed to load model!".format(self.name))
                else:
                    raise
                data['time_model_load'] += time.time() - tic

                # Aggregate / reduce (incremental average)
                # TODO: extend with metadata from client
                # Need to know round id of model update, present round id, number of data points in the update, etc.
                tic = time.time()
                if nr_processed_models == 0:
                    model = model_next
                else:
                    model = helper.increment_average(
                        model, model_next, nr_processed_models + 1)
                data['time_model_aggregation'] += time.time() - tic

                nr_processed_models += 1
                self.model_updates.task_done()
            except queue.Empty:
                self.server.report_status("AGGREGATOR({}): combining model updates: {} completed.".format(self.name,
                                                                                                          nr_processed_models,
                                                                                                          max_nr_models))
                time.sleep(polling_interval)
                round_time += polling_interval
            except Exception as e:
                self.server.report_status(
                    "AGGERGATOR({}): Error encoutered while reading model update, skipping this update. {}".format(self.name, e))
                max_nr_models -= 1
                if max_nr_models <= 0:
                    return None, data
                self.model_updates.task_done()

            if round_time >= time_window:
                self.server.report_status("AGGREGATOR({}): aggregation round completed time window.".format(
                    self.name), log_level=fedn.Status.WARNING)
                # TODO: Generalize policy for what to do in case of timeout.
                break

        data['nr_processed_models'] = nr_processed_models

        self.server.report_status("AGGREGATOR({}): Round completed, aggregated {} models.".format(self.name, nr_processed_models),
                                  log_level=fedn.Status.INFO)
        return model, data
