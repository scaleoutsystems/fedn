import base64

from fedn.network.combiner.interfaces import (CombinerInterface,
                                              CombinerUnavailableError)
from fedn.network.loadbalancer.leastpacked import LeastPacked


class Network:
    """ FEDn network. """

    def __init__(self, control, statestore, load_balancer=None):
        """ """
        self.statestore = statestore
        self.control = control
        self.id = statestore.network_id

        if not load_balancer:
            self.load_balancer = LeastPacked(self)
        else:
            self.load_balancer = load_balancer

    def get_combiner(self, name):
        """

        :param name:
        :return:
        """
        combiners = self.get_combiners()
        for combiner in combiners:
            if name == combiner.name:
                return combiner
        return None

    def get_combiners(self):
        """

        :return:
        """
        data = self.statestore.get_combiners()
        combiners = []
        for c in data:
            if c['certificate']:
                cert = base64.b64decode(c['certificate'])
                key = base64.b64decode(c['key'])
            else:
                cert = None
                key = None

            combiners.append(
                CombinerInterface(c['parent'], c['name'], c['address'], c['fqdn'], c['port'],
                                  certificate=cert, key=key, ip=c['ip']))

        return combiners

    def add_combiner(self, combiner):
        """

        :param combiner:
        :return:
        """
        if not self.control.idle():
            print("Reducer is not idle, cannot add additional combiner.")
            return

        if self.get_combiner(combiner.name):
            return

        print("adding combiner {}".format(combiner.name), flush=True)
        self.statestore.set_combiner(combiner.to_dict())

    def remove_combiner(self, combiner):
        """

        :param combiner:
        :return:
        """
        if not self.control.idle():
            print("Reducer is not idle, cannot remove combiner.")
            return
        self.statestore.delete_combiner(combiner.name)

    def find_available_combiner(self):
        """

        :return:
        """
        combiner = self.load_balancer.find_combiner()
        return combiner

    def handle_unavailable_combiner(self, combiner):
        """ This callback is triggered if a combiner is found to be unresponsive. """
        # TODO: Implement strategy to handle an unavailable combiner.
        print("REDUCER CONTROL: Combiner {} unavailable.".format(
            combiner.name), flush=True)

    def add_client(self, client):
        """ Add a new client to the network.

        :param client:
        :return:
        """

        if self.get_client(client['name']):
            return

        print("adding client {}".format(client['name']), flush=True)
        self.statestore.set_client(client)

    def get_client(self, name):
        """

        :param name:
        :return:
        """
        ret = self.statestore.get_client(name)
        return ret

    def update_client_data(self, client_data, status, role):
        """ Update client status on DB"""
        self.statestore.update_client_status(client_data, status, role)

    def get_client_info(self):
        """ list available client in DB"""
        return self.statestore.list_clients()

    def describe(self):
        """ """
        network = []
        for combiner in self.get_combiners():
            try:
                network.append(combiner.report())
            except CombinerUnavailableError:
                # TODO, do better here.
                pass
        return network

    def check_health(self):
        """ """
        pass
