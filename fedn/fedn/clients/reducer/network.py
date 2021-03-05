import copy
import time
import base64

from .state import ReducerState
from fedn.clients.reducer.interfaces import CombinerInterface
from fedn.clients.reducer.interfaces import CombinerUnavailableError


class Network: 
    """ FEDn network. """

    def __init__(self,control, statestore):
        """ """ 
        self.statestore = statestore
        self.control = control
        self.id = statestore.network_id

    @classmethod
    def from_statestore(self,network_id):
        """ """

    def get_combiner(self,name):
        return self.statestore.get_combiner(name)

    def get_combiners(self):
        # TODO: Read in combiners from statestore
        data = self.statestore.get_combiners()
        combiners=[]
        for c in data:
            combiners.append(CombinerInterface(c['parent'],c['name'],c['address'],c['port'],base64.b64decode(c['certificate']),base64.b64decode(c['key']),c['ip']))

        return combiners

    def add_combiner(self, combiner):
        if not self.control.idle():
            print("Reducer is not idle, cannot add additional combiner.")
            return

        if self.find(combiner.name):
            return

        print("adding combiner {}".format(combiner.name), flush=True)
        self.statestore.set_combiner(combiner.to_dict())

    def add_client(self,client):
        if not self.control.idle():
            print("Reducer is not idle, cannot add additional client.")
            return

        if self.find_client(client['name']):
            return

        print("adding client {}".format(client['name']), flush=True)
        self.statestore.set_client(client)

    def remove(self, combiner):
        if not self.control.idle():
            print("Reducer is not idle, cannot remove combiner.")
            return
        self.statestore.delete_combiner(combiner.name)

    def find(self, name):
        combiners = self.get_combiners()
        for combiner in combiners:
            if name == combiner.name:
                return combiner
        return None

    def find_client(self,name):
        ret = self.statestore.get_client(name)
        return ret
            
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
