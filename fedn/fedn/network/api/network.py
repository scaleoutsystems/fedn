import base64

from fedn.network.combiner.interfaces import (CombinerInterface,
                                              CombinerUnavailableError)
from fedn.network.loadbalancer.leastpacked import LeastPacked

__all__ = 'Network',


class Network:
    """ FEDn network interface. This class is used to interact with the network.
        Note: This class contain redundant code, which is not used in the current version of FEDn.
        Some methods has been moved to :class:`fedn.network.api.interface.API`.
         """

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
        """ Get combiner by name.

        :param name: name of combiner
        :type name: str
        :return: The combiner instance object
        :rtype: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        """
        combiners = self.get_combiners()
        for combiner in combiners:
            if name == combiner.name:
                return combiner
        return None

    def get_combiners(self):
        """ Get all combiners in the network.

        :return: list of combiners objects
        :rtype: list(:class:`fedn.network.combiner.interfaces.CombinerInterface`)
        """
        data = self.statestore.get_combiners()
        combiners = []
        for c in data["result"]:
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
        """ Add a new combiner to the network.

        :param combiner: The combiner instance object
        :type combiner: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        :return: None
        """
        if not self.control.idle():
            print("Reducer is not idle, cannot add additional combiner.")
            return

        if self.get_combiner(combiner.name):
            return

        print("adding combiner {}".format(combiner.name), flush=True)
        self.statestore.set_combiner(combiner.to_dict())

    def remove_combiner(self, combiner):
        """ Remove a combiner from the network.

        :param combiner: The combiner instance object
        :type combiner: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        :return: None
        """
        if not self.control.idle():
            print("Reducer is not idle, cannot remove combiner.")
            return
        self.statestore.delete_combiner(combiner.name)

    def find_available_combiner(self):
        """ Find an available combiner in the network.

        :return: The combiner instance object
        :rtype: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        """
        combiner = self.load_balancer.find_combiner()
        return combiner

    def handle_unavailable_combiner(self, combiner):
        """ This callback is triggered if a combiner is found to be unresponsive.

        :param combiner: The combiner instance object
        :type combiner: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        :return: None
        """
        # TODO: Implement strategy to handle an unavailable combiner.
        print("REDUCER CONTROL: Combiner {} unavailable.".format(
            combiner.name), flush=True)

    def add_client(self, client):
        """ Add a new client to the network.

        :param client: The client instance object
        :type client: dict
        :return: None
        """

        if self.get_client(client['name']):
            return

        print("adding client {}".format(client['name']), flush=True)
        self.statestore.set_client(client)

    def get_client(self, name):
        """ Get client by name.

        :param name: name of client
        :type name: str
        :return: The client instance object
        :rtype: ObjectId
        """
        ret = self.statestore.get_client(name)
        return ret

    def update_client_data(self, client_data, status, role):
        """ Update client status in statestore.

        :param client_data: The client instance object
        :type client_data: dict
        :param status: The client status
        :type status: str
        :param role: The client role
        :type role: str
        :return: None
        """
        self.statestore.update_client_status(client_data, status, role)

    def get_client_info(self):
        """ list available client in statestore.

        :return: list of client objects
        :rtype: list(ObjectId)
        """
        return self.statestore.list_clients()

    def describe(self):
        """ Describe the network.

        :return: The network description
        :rtype: dict
        """
        network = []
        for combiner in self.get_combiners():
            try:
                network.append(combiner.report())
            except CombinerUnavailableError:
                # TODO, do better here.
                pass
        return network
