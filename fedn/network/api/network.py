import os
from typing import List

from fedn.common.log_config import logger
from fedn.network.combiner.interfaces import CombinerInterface
from fedn.network.loadbalancer.leastpacked import LeastPacked
from fedn.network.storage.dbconnection import DatabaseConnection

__all__ = ("Network",)


class Network:
    """FEDn network interface. This class is used to interact with the network.
    Note: This class contain redundant code, which is not used in the current version of FEDn.
    Some methods has been moved to :class:`fedn.network.api.interface.API`.
    """

    def __init__(self, control, network_id: str, dbconn: DatabaseConnection, load_balancer=None):
        """ """
        self.control = control
        self.id = network_id
        self.db = dbconn

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

    def get_combiners(self) -> List[CombinerInterface]:
        """Get all combiners in the network.

        :return: list of combiners objects
        :rtype: list(:class:`fedn.network.combiner.interfaces.CombinerInterface`)
        """
        result = self.db.combiner_store.list(limit=0, skip=0, sort_key=None)
        combiners = []
        for combiner in result:
            name = combiner.name.upper()
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
            combiners.append(
                CombinerInterface(combiner.parent, combiner.name, combiner.address, combiner.fqdn, combiner.port, certificate=cert, ip=combiner.ip)
            )

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
