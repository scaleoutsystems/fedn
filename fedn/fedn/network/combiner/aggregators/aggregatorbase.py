import importlib
import json
import queue
from abc import ABC, abstractmethod

import fedn.common.net.grpc.fedn_pb2 as fedn

AGGREGATOR_PLUGIN_PATH = "fedn.network.combiner.aggregators.{}"


class AggregatorBase(ABC):
    """ Abstract class defining an aggregator. 

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

    @abstractmethod
    def __init__(self, storage, server, modelservice, control):
        """ Initialize the aggregator."""
        self.name = self.__class__.__name__
        self.storage = storage
        self.server = server
        self.modelservice = modelservice
        self.control = control
        self.model_updates = queue.Queue()

    @abstractmethod
    def combine_models(self, nr_expected_models=None, nr_required_models=1, helper=None, timeout=180, delete_models=True):
        """Routine for combining model updates. Implemented in subclass.

        :param nr_expected_models: Number of expected models. If None, wait for all models.
        :type nr_expected_models: int
        :param nr_required_models: Number of required models to combine.
        :type nr_required_models: int
        :param helper: A helper object.
        :type helper: :class: `fedn.utils.plugins.helperbase.HelperBase`
        :param timeout: Timeout in seconds to wait for models to be combined.
        :type timeout: int
        :param delete_models: Delete client models after combining.
        :type delete_models: bool
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
            self.server.report_status("AGGREGATOR({}): Failed to receive model update! {}".format(self.name, e),
                                      log_level=fedn.Status.WARNING)
            pass

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

    def get_state(self):
        """ Get the state of the aggregator's queue, including the number of model updates."""
        state = {
            'queue_len': self.model_updates.qsize()

        }
        return state


def get_aggregator(aggregator_module_name, storage, server, modelservice, control):
    """ Return an instance of the helper class.

    :param helper_module_name: The name of the helper plugin module.
    :type helper_module_name: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    :param control: A handle to the :class: `fedn.network.combiner.round.RoundController`
    :type control: class: `fedn.network.combiner.round.RoundController`
    :return: An aggregator instance.
    :rtype: class: `fedn.combiner.aggregators.AggregatorBase`
    """
    aggregator_plugin = AGGREGATOR_PLUGIN_PATH.format(aggregator_module_name)
    aggregator = importlib.import_module(aggregator_plugin)
    return aggregator.Aggregator(storage, server, modelservice, control)
