import queue
from abc import ABC, abstractmethod

import fedn.common.net.grpc.fedn_pb2 as fedn


class ModelUpdateError(Exception):
    pass


class AggregatorBase(ABC):
    """ Abstract class defining helpers. """

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
        pass

    def on_model_update(self, model_update):
        """Callback when a new model update is recieved from a client.
            Performs (optional) pre-processing and then puts the update id
            on the aggregation queue.

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
        """ Callback when a new model validation is recieved from a client.
            Performs (optional) pre-processing and writes the validation
            to the database.

        :param validation: Dict containing validation data sent by client.
                           Must be valid JSON.
        :type validation: dict
        """

        # self.report_validation(validation)
        self.server.report_status("AGGREGATOR({}): callback processed validation {}".format(self.name, model_validation.model_id),
                                  log_level=fedn.Status.INFO)

    def load_model_update(self, helper, model_id):
        """Read model update from file.

        :param helper: An instance of :class: `fedn.utils.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.HelperBase`
        :param model_id: The ID of the model update, UUID in str format  
        :type model_id: str
        """

        model_str = self.control.load_model_str(model_id)
        if model_str:
            try:
                model = helper.load_model_from_BytesIO(model_str.getbuffer())
            except IOError:
                self.server.report_status(
                    "AGGREGATOR({}): Failed to load model!".format(self.name))
        else:
            raise ModelUpdateError("Failed to load model.")

        return model
