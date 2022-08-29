from abc import ABC, abstractmethod


class LoadBalancerBase(ABC):

    def __init__(self, network):
        """ """
        self.network = network

    @abstractmethod
    def find_combiner(self):
        """ """
        pass
