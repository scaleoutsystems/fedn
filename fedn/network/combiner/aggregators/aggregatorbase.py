import importlib
from abc import ABC, abstractmethod

from fedn.network.combiner.updatehandler import UpdateHandler

AGGREGATOR_PLUGIN_PATH = "fedn.network.combiner.aggregators.{}"


class AggregatorBase(ABC):
    """Abstract class defining an aggregator.

    :param control: A handle to the :class: `fedn.network.combiner.updatehandler.UpdateHandler`
    :type control: class: `fedn.network.combiner.updatehandler.UpdateHandler`
    """

    @abstractmethod
    def __init__(self, update_handler: UpdateHandler):
        """Initialize the aggregator."""
        self.name = self.__class__.__name__
        self.update_handler = update_handler

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


def get_aggregator(aggregator_module_name, update_handler):
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
    return aggregator.Aggregator(update_handler)
