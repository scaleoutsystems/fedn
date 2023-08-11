from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.loadbalancer.loadbalancerbase import LoadBalancerBase


class LeastPacked(LoadBalancerBase):
    """ Load balancer that selects the combiner with the least number of attached clients.

    :param network: A handle to the network.
    :type network: class: `fedn.network.api.network.Network`
    """

    def __init__(self, network):
        super().__init__(network)

    def find_combiner(self):
        """
            Find the combiner with the least number of attached clients.

        """
        min_clients = None
        selected_combiner = None

        for combiner in self.network.get_combiners():
            try:
                if combiner.allowing_clients():
                    combiner_state = combiner.report()
                    if not min_clients:
                        min_clients = combiner_state['nr_active_clients']
                        selected_combiner = combiner
                    elif combiner_state['nr_active_clients'] < min_clients:
                        min_clients = combiner_state['nr_active_clients']
                        selected_combiner = combiner
            except CombinerUnavailableError:
                pass

        return selected_combiner
