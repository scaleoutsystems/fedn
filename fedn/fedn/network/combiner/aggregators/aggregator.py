import json
import queue
from abc import ABC, abstractmethod

import fedn.common.net.grpc.fedn_pb2 as fedn


class Aggregator(ABC):
    """ Abstract class defining an aggregator. """

    @abstractmethod
    def __init__(self, id, storage, server, modelservice, control):
        """ """
        self.name = ""
        self.storage = storage
        self.id = id
        self.server = server
        self.modelservice = modelservice
        self.control = control
        self.model_updates = queue.Queue()

    @abstractmethod
    def combine_models(self, nr_expected_models=None, nr_required_models=1, helper=None, timeout=180):
        """Routine for combining model updates. Implemented in subclass. """
        pass

    def on_model_update(self, model_update):
        """Callback when a new client model update is recieved.
           Performs (optional) pre-processing and then puts the update id
           on the aggregation queue. Override in subclass as needed. 

        :param model_update: A ModelUpdate message.
        :type model_id: str
        """
        try:
            self.server.report_status("AGGREGATOR({}): callback received model update {}".format(self.name, model_update.model_update_id),
                                      log_level=fedn.Status.INFO)

            # Push the model update to the processing queue
            self.model_updates.put(model_update)
        except Exception as e:
            self.server.report_status("AGGREGATOR({}): Failed to receive candidate model! {}".format(self.name, e),
                                      log_level=fedn.Status.WARNING)
            pass

    def on_model_validation(self, model_validation):
        """ Callback when a new client model validation is recieved.
            Performs (optional) pre-processing and then writes the validation
            to the database. Override in subclass as needed.

        :param validation: Dict containing validation data sent by client.
                           Must be valid JSON.
        :type validation: dict
        """

        # self.report_validation(validation)
        self.server.report_status("AGGREGATOR({}): callback processed validation {}".format(self.name, model_validation.model_id),
                                  log_level=fedn.Status.INFO)

    def next_model_update(self, helper):
        """ """
        model_update = self.model_updates.get(block=False)
        model_id = model_update.model_update_id
        model_next = self.control.load_model_update(helper, model_id)
        # Get relevant metadata
        data = json.loads(model_update.meta)['training_metadata']
        config = json.loads(json.loads(model_update.meta)['config'])
        data['round_id'] = config['round_id']

        return model_next, data, model_id
