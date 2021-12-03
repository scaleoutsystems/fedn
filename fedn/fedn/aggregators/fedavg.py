import json
import os
import queue
import tempfile
import time
import uuid
import sys

import fedn.common.net.grpc.fedn_pb2 as fedn
from fedn.utils.helpers import get_helper
from fedn.aggregators.aggregator import AggregatorBase

class FedAvgAggregator(AggregatorBase):
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

        super().__init__(id,storage, server, modelservice, control)
        
        self.name = "FedAvg"
        self.validations = {}
        self.model_updates = queue.Queue()

    def on_model_update(self, model_id):
        """Callback when a new model update is recieved from a client.
            Performs (optional) pre-processing and the puts the update id
            on the aggregation queue. 
        
        :param model_id: ID of model update
        :type model_id: str
        """
        try:
            self.server.report_status("AGGREGATOR({}): callback received model {}".format(self.name, model_id),
                               log_level=fedn.Status.INFO)

            # Push the model update to the processing queue
            self.model_updates.put(model_id)
        except Exception as e:
            self.server.report_status("AGGREGATOR({}): Failed to receive candidate model! {}".format(self.name, e),
                               log_level=fedn.Status.WARNING)
            pass

    def on_model_validation(self, validation):
        """ Callback when a new model validation is recieved from a client. 

        :param validation: Dict containing validation data sent by client. 
                           Must be valid JSON.
        :type validation: dict
        """

        # Currently, the validations are actually sent as status messages 
        # directly in the client, so here we are just storing them in the
        # combiner memory. This will need to be refactored later so that this 
        # callback is responsible for reporting the validation to the db. 

        model_id = validation.model_id
        data = json.loads(validation.data)
        try:
            self.validations[model_id].append(data)
        except KeyError:
            self.validations[model_id] = [data]

        self.server.report_status("AGGREGATOR({}): callback processed validation {}".format(self.name, validation.model_id),
                           log_level=fedn.Status.INFO)


    def combine_models(self, nr_expected_models=None, nr_required_models=1, helper=None, timeout=180):
        """Compute a running average of model updates.

        :param nr_expected_models: The number of updates expected in this round, defaults to None
        :type nr_expected_models: int, optional
        :param nr_required_models: The number of updates needed to a valid round, defaults to 1
        :type nr_required_models: int, optional
        :param helper: An instance of :class: `fedn.utils.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.HelperBase`, optional
        :param timeout: Timeout for model updates, defaults to 180
        :type timeout: int, optional
        :return: The global model and metadata
        :rtype: tuple
        """

        
        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0

        self.server.report_status("AGGREGATOR({}): Aggregating model updates...".format(self.name))

        round_time = 0.0
        polling_interval = 1.0
        nr_processed_models = 0
        while nr_processed_models < nr_expected_models:
            try:
                model_id = self.model_updates.get(block=False)
                self.server.report_status("AGGREGATOR({}): Received model update with id {}".format(self.name, model_id))

                # Load the model update from disk
                tic = time.time()
                model_str = self.control.load_model_fault_tolerant(model_id)
                if model_str:
                    try:
                        model_next = helper.load_model_from_BytesIO(model_str.getbuffer())
                    except IOError:
                        self.server.report_status("AGGREGATOR({}): Failed to load model!".format(self.name))
                else: 
                    raise
                data['time_model_load'] += time.time() - tic

                # Aggregate / reduce 
                tic = time.time()
                if nr_processed_models == 0:
                    model = model_next
                else:
                    model = helper.increment_average(model, model_next, nr_processed_models + 1)
                data['time_model_aggregation'] += time.time() - tic

                nr_processed_models += 1
                self.model_updates.task_done()
            except queue.Empty:
                self.server.report_status("AGGREGATOR({}): waiting for model updates: {} of {} completed.".format(self.name, 
                                                                                                                  nr_processed_models,
                                                                                                                  nr_expected_models))
                time.sleep(polling_interval)
                round_time += polling_interval
            except Exception as e:
                self.server.report_status("AGGERGATOR({}): Error encoutered while reading model update, skipping this update. {}".format(self.name, e))
                nr_expected_models -= 1
                if nr_expected_models <= 0:
                    return None, data
                self.model_updates.task_done()
           
            if round_time >= timeout:
                self.server.report_status("AGGREGATOR({}): training round timed out.".format(self.name), log_level=fedn.Status.WARNING)
                # TODO: Generalize policy for what to do in case of timeout. 
                if nr_processed_models >= nr_required_models:
                    break
                else:
                    return None, data

        data['nr_successful_updates'] = nr_processed_models

        self.server.report_status("AGGREGATOR({}): Training round completed, aggregated {} models.".format(self.name, nr_processed_models),
                           log_level=fedn.Status.INFO)
        return model, data
