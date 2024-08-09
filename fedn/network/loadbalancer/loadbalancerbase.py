from abc import ABC, abstractmethod


class LoadBalancerBase(ABC):
    """Abstract base class for load balancers.

    :param network: A handle to the network.
    :type network: class: `fedn.network.api.network.Network`
    """

    def __init__(self, network):
        """ """
        self.network = network

    @abstractmethod
    def find_combiner(self):
        """Find a combiner to connect to."""
        pass
