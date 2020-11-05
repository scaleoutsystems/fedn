import copy
import time

from .state import ReducerState
from fedn.clients.reducer.interfaces import CombinerUnavailableError


class Network: 
    """ FEDn network. """

    def __init__(self,control, statestore):
        """ """ 
        self.statestore = statestore
        self.control = control
        self.combiners = []
        self.id = statestore.network_id

    @classmethod
    def from_statestore(self,network_id):
        """ """

    def get_combiners(self):
        return self.combiners

    def add_combiner(self, combiner):
        if not self.control.idle():
            print("Reducer is not idle, cannot add additional combiner.")
            return

        if self.find(combiner.name):
            return

        print("adding combiner {}".format(combiner.name), flush=True)
        self.statestore.set_combiner(combiner.to_dict())
        self.combiners.append(combiner)

    def add_client(self,client):
        if not self.control.idle():
            print("Reducer is not idle, cannot add additional client.")
            return

        if self.find_client(client['name']):
            return

        print("adding client {}".format(client.name), flush=True)
        self.statestore.set_client(client)

    def remove(self, combiner):
        if not self.control.idle():
            print("Reducer is not idle, cannot remove combiner.")
            return
        self.combiners.remove(combiner)

    def find(self, name):
        for combiner in self.combiners:
            if name == combiner.name:
                return combiner
        return None

    def find_client(self,name):
        ret = self.statestore.get_client(name)
        return ret
            
    def describe(self):
        """ """
        network = []
        for combiner in self.combiners:
            try:
                network.append(combiner.report())
            except CombinerUnavailableError:
                # TODO, do better here.
                pass
        return network

    def check_health(self):
        """ """
        pass
