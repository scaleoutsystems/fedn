import os

from fedn.common.log_config import logger
from fedn.network.combiner.interfaces import CombinerInterface
from fedn.network.loadbalancer.leastpacked import LeastPacked
from fedn.network.storage.statestore.stores.client_store import ClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore

__all__ = ("Network",)


class Network:
    """FEDn network interface. This class is used to interact with the network.
    Note: This class contain redundant code, which is not used in the current version of FEDn.
    Some methods has been moved to :class:`fedn.network.api.interface.API`.
    """

    def __init__(self, control, network_id: str, combiner_store: CombinerStore, client_store: ClientStore, load_balancer=None):
        """ """
        self.combiner_store = combiner_store
        self.client_store = client_store
        self.control = control
        self.id = network_id

        if not load_balancer:
            self.load_balancer = LeastPacked(self)
        else:
            self.load_balancer = load_balancer

    def get_combiner(self, name):
        """Get combiner by name.

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
        """Get all combiners in the network.

        :return: list of combiners objects
        :rtype: list(:class:`fedn.network.combiner.interfaces.CombinerInterface`)
        """
        data = self.combiner_store.list(limit=0, skip=0, sort_key=None)
        combiners = []
        for c in data["result"]:
            name = c["name"].upper()
            # General certificate handling, same for all combiners.
            if os.environ.get("FEDN_GRPC_CERT_PATH"):
                with open(os.environ.get("FEDN_GRPC_CERT_PATH"), "rb") as f:
                    cert = f.read()
            # Specific certificate handling for each combiner.
            elif os.environ.get(f"FEDN_GRPC_CERT_PATH_{name}"):
                cert_path = os.environ.get(f"FEDN_GRPC_CERT_PATH_{name}")
                with open(cert_path, "rb") as f:
                    cert = f.read()
            else:
                cert = None
            combiners.append(CombinerInterface(c["parent"], c["name"], c["address"], c["fqdn"], c["port"], certificate=cert, ip=c["ip"]))

        return combiners

    def find_available_combiner(self):
        """Find an available combiner in the network.

        :return: The combiner instance object
        :rtype: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        """
        combiner = self.load_balancer.find_combiner()
        return combiner

    def handle_unavailable_combiner(self, combiner):
        """This callback is triggered if a combiner is found to be unresponsive.

        :param combiner: The combiner instance object
        :type combiner: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        :return: None
        """
        # TODO: Implement strategy to handle an unavailable combiner.
        logger.warning("REDUCER CONTROL: Combiner {} unavailable.".format(combiner.name))

    def add_client(self, client):
        """Add a new client to the network.

        :param client: The client instance object
        :type client: dict
        :return: None
        """
        if self.get_client(client["client_id"]):
            return

        logger.info("adding client {}".format(client["client_id"]))
        self.client_store.upsert(client)

    def get_client(self, client_id: str):
        """Get client by client_id.

        :param client_id: client_id of client
        :type client_id: str
        :return: The client instance object
        :rtype: ObjectId
        """
        try:
            client = self.client_store.get(client_id)
            return client
        except Exception:
            return None

    def get_client_info(self):
        """List available client in statestore.

        :return: list of client objects
        :rtype: list(ObjectId)
        """
        return self.client_store.list(limit=0, skip=0, sort_key=None)
