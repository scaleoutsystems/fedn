import importlib
import json
import queue
import traceback
from abc import ABC, abstractmethod

from fedn.common.log_config import logger

AGGREGATOR_PLUGIN_PATH = "fedn.network.combiner.aggregators.{}"


class AggregatorBase(ABC):
    """Abstract class defining an aggregator.

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

    @abstractmethod
    def __init__(self, storage, server, modelservice, round_handler):
        """Initialize the aggregator."""
        self.name = self.__class__.__name__
        self.storage = storage
        self.server = server
        self.modelservice = modelservice
        self.round_handler = round_handler
        self.model_updates = queue.Queue()

    @abstractmethod
    def combine_models(self, nr_expected_models=None, nr_required_models=1, helper=None, timeout=180, delete_models=True, parameters=None):
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
        :param parameters: Additional key-word arguments.
        :type parameters: dict
        :return: The global model and metadata
        :rtype: tuple
        """
        pass

    def on_model_update(self, model_update):
        """Callback when a new client model update is recieved.

        Performs (optional) validation and pre-processing,
        and then puts the update id on the aggregation queue.
        Override in subclass as needed.

        :param model_update: fedn.network.grpc.fedn.proto.ModelUpdate
        :type model_id: str
        """
        try:
            logger.info("AGGREGATOR({}): callback received model update {}".format(self.name, model_update.model_update_id))

            # Validate the update and metadata
            valid_update = self._validate_model_update(model_update)
            if valid_update:
                # Push the model update to the processing queue
                self.model_updates.put(model_update)
            else:
                logger.warning("AGGREGATOR({}): Invalid model update, skipping.".format(self.name))
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("AGGREGATOR({}): failed to receive model update: {}".format(self.name, e))
            logger.error(tb)
            pass

    def _validate_model_update(self, model_update):
        """Validate the model update.

        :param model_update: A ModelUpdate message.
        :type model_update: object
        :return: True if the model update is valid, False otherwise.
        :rtype: bool
        """
        try:
            data = json.loads(model_update.meta)["training_metadata"]
            _ = data["num_examples"]
        except KeyError:
            tb = traceback.format_exc()
            logger.error("AGGREGATOR({}): Invalid model update, missing metadata.".format(self.name))
            logger.error(tb)
            return False
        return True

    def next_model_update(self):
        """Get the next model update from the queue.

        :param helper: A helper object.
        :type helper: object
        :return: The model update.
        :rtype: fedn.network.grpc.fedn.proto.ModelUpdate
        """
        model_update = self.model_updates.get(block=False)
        return model_update

    def load_model_update(self, model_update, helper):
        """Load the memory representation of the model update.

        Load the model update paramters and the
        associate metadata into memory.

        :param model_update: The model update.
        :type model_update: fedn.network.grpc.fedn.proto.ModelUpdate
        :param helper: A helper object.
        :type helper: fedn.utils.helpers.helperbase.Helper
        :return: A tuple of (parameters, metadata)
        :rtype: tuple
        """
        model_id = model_update.model_update_id
        model = self.round_handler.load_model_update(helper, model_id)
        # Get relevant metadata
        metadata = json.loads(model_update.meta)
        if "config" in metadata.keys():
            # Used in Python client
            config = json.loads(metadata["config"])
        else:
            # Used in C++ client
            config = json.loads(model_update.config)
        training_metadata = metadata["training_metadata"]
        training_metadata["round_id"] = config["round_id"]

        return model, training_metadata

    def get_state(self):
        """Get the state of the aggregator's queue, including the number of model updates."""
        state = {"queue_len": self.model_updates.qsize()}
        return state


def get_aggregator(aggregator_module_name, storage, server, modelservice, control):
    """Return an instance of the helper class.

    :param helper_module_name: The name of the helper plugin module.
    :type helper_module_name: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    :param control: A handle to the :class: `fedn.network.combiner.roundhandler.RoundHandler`
    :type control: class: `fedn.network.combiner.roundhandler.RoundHandler`
    :return: An aggregator instance.
    :rtype: class: `fedn.combiner.aggregators.AggregatorBase`
    """
    aggregator_plugin = AGGREGATOR_PLUGIN_PATH.format(aggregator_module_name)
    aggregator = importlib.import_module(aggregator_plugin)
    return aggregator.Aggregator(storage, server, modelservice, control)
