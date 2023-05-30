import json
import queue
from abc import ABC, abstractmethod

import fedn.common.net.grpc.fedn_pb2 as fedn


class Aggregator(ABC):
    """ Abstract class defining an aggregator. """

    @abstractmethod
    def __init__(self, id, storage, server, modelservice, control):
        """ Initialize the aggregator.

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
        self.name = self.__class__.__name__
        self.storage = storage
        self.id = id
        self.server = server
        self.modelservice = modelservice
        self.control = control
        self.model_updates = queue.Queue()
        # Track the number of model validations performed
        self.validations = {}

    @abstractmethod
    def combine_models(self, nr_expected_models=None, nr_required_models=1, helper=None, timeout=180):
        """Routine for combining model updates. Implemented in subclass.

        :param nr_expected_models: Number of expected models. If None, wait for all models.
        :type nr_expected_models: int
        :param nr_required_models: Number of required models to combine.
        :type nr_required_models: int
        :param helper: A helper object.
        :type helper: :class: `fedn.utils.plugins.helperbase.HelperBase`
        :param timeout: Timeout in seconds to wait for models to be combined.
        :type timeout: int
        :return: A combined model.
        """
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

            # Validate the update and metadata
            valid_update = self._validate_model_update(model_update)
            if valid_update:
                # Push the model update to the processing queue
                self.model_updates.put(model_update)
            else:
                self.server.report_status("AGGREGATOR({}): Invalid model update, skipping.".format(self.name))
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

        self.report_validation(model_validation)
        self.server.report_status("AGGREGATOR({}): callback processed validation {}".format(self.name, model_validation.model_id),
                                  log_level=fedn.Status.INFO)
        #total_expected_validations = self.server.nr_active_validators()
        # Check if all validations have been received for the model and delete the model if so
        # if total_expected_validations == self.get_total_validations(model_validation.model_id):
        #    self.server.report_status("AGGREGATOR({}): All validations received for model {}, deleting model.".format(self.name, model_validation.model_id),
        #                              log_level=fedn.Status.INFO)
        #    self.modelservice.models.delete(model_validation.model_id)
        # Delete the model from the validation dictionary
        # del self.validations[model_validation.model_id]

    def report_validation(self, request):
        """ Report validation to dict.

        :param request: A validation request.
        :type request: object
        """
        client_name = request.sender.name
        model_id = request.model_id
        if model_id not in self.validations.keys():
            self.validations[model_id] = [client_name]
        else:
            self.validations[model_id].append(client_name)

    # Get total number of validations for a model
    def get_total_validations(self, model_id):
        """ Get total number of validations for a model.

        :param model_id: A model id.
        :type model_id: str
        :return: The total number of validations.
        :rtype: int
        """
        if model_id not in self.validations.keys():
            return 0
        else:
            return len(self.validations[model_id])

    def _validate_model_update(self, model_update):
        """ Validate the model update.

        :param model_update: A ModelUpdate message.
        :type model_update: object
        :return: True if the model update is valid, False otherwise.
        :rtype: bool
        """
        # TODO: Validate the metadata to check that it contains all variables assumed by the aggregator.
        data = json.loads(model_update.meta)['training_metadata']
        if 'num_examples' not in data.keys():
            self.server.report_status("AGGREGATOR({}): Model validation failed, num_examples missing in metadata.".format(self.name))
            return False
        return True

    def next_model_update(self, helper):
        """ Get the next model update from the queue.

        :param helper: A helper object.
        :type helper: object
        :return: A tuple containing the model update, metadata and model id.
        :rtype: tuple
        """
        model_update = self.model_updates.get(block=False)
        model_id = model_update.model_update_id
        model_next = self.control.load_model_update(helper, model_id)
        # Get relevant metadata
        data = json.loads(model_update.meta)['training_metadata']
        config = json.loads(json.loads(model_update.meta)['config'])
        data['round_id'] = config['round_id']

        return model_next, data, model_id
