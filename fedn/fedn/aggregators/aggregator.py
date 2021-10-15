import collections
from abc import ABC, abstractmethod
import os
import tempfile


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

    @abstractmethod
    def on_model_update(self, model_id):
        pass

    @abstractmethod
    def on_model_validation(self, validation):
        pass

    @abstractmethod
    def combine_models(self, nr_expected_models=None, nr_required_models=1, helper=None, timeout=180):
        pass
   
  
#def get_aggregator(aggregator_type):
#    """ Return an instance of the aggregator class. 
#
#    :param aggregator_type (str): The aggregator type ('fedavg')
#    :return: 
#    """
#    if helper_type == 'fedavg':
#        from fedn.aggregators.fedavg import FedAvgAggregator
#        return FedAvgAggregator()