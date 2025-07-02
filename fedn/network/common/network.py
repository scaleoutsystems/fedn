import os
from typing import List

import fedn
from fedn.common.log_config import logger
from fedn.network.common.interfaces import CombinerInterface, CombinerUnavailableError, ControlInterface
from fedn.network.controller.shared import MisconfiguredHelper
from fedn.network.loadbalancer.leastpacked import LeastPacked
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.stores.dto.model import ModelDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

__all__ = ("Network",)


class Network:
    """FEDn network interface. This class is used to interact with the database in a consistent way accross different containers."""

    def __init__(self, dbconn: DatabaseConnection, repository: Repository, load_balancer=None, controller_host: str = None, controller_port: int = None):
        """ """
        self.db = dbconn
        self.repository = repository

        if not load_balancer:
            self.load_balancer = LeastPacked(self)
        else:
            self.load_balancer = load_balancer

        self.controller = self._init_controller_interface(controller_host, controller_port)

    def _init_controller_interface(self, host, port) -> ControlInterface:
        """Get a control instance from global config.

        :return: ControlInterface instance.
        :rtype: :class:`fedn.network.common.interfaces.ControlInterface`
        """
        cert = None
        name = "CONTROL".upper()
        # General certificate handling, same for all combiners.
        if os.environ.get("FEDN_GRPC_CERT_PATH"):
            with open(os.environ.get("FEDN_GRPC_CERT_PATH"), "rb") as f:
                cert = f.read()
        # Specific certificate handling for each combiner.
        elif os.environ.get(f"FEDN_GRPC_CERT_PATH_{name}"):
            cert_path = os.environ.get(f"FEDN_GRPC_CERT_PATH_{name}")
            with open(cert_path, "rb") as f:
                cert = f.read()

        # TODO: Remove hardcoded values
        return ControlInterface(host, port, cert)

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
                CombinerInterface(
                    combiner.combiner_id, combiner.parent, combiner.name, combiner.address, combiner.fqdn, combiner.port, certificate=cert, ip=combiner.ip
                )
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

    def get_control(self) -> ControlInterface:
        """Get a control instance from global config.

        :return: ControlInterface instance.
        :rtype: :class:`fedn.network.common.interfaces.ControlInterface`
        """
        return self.controller

    def get_helper(self):
        """Get a helper instance from global config.

        :return: Helper instance.
        :rtype: :class:`fedn.utils.plugins.helperbase.HelperBase`
        """
        helper_type: str = None

        try:
            active_package = self.db.package_store.get_active()
            helper_type = active_package.helper
        except Exception:
            logger.error("Failed to get active helper")

        helper = fedn.utils.helpers.helpers.get_helper(helper_type)
        if not helper:
            raise MisconfiguredHelper("Unsupported helper type {}, please configure compute_package.helper !".format(helper_type))
        return helper

    def commit_model(self, model: dict = None, session_id: str = None, name: str = None) -> str:
        """Commit a model to the global model trail. The model commited becomes the lastest consensus model.

        :param model_id: Unique identifier for the model to commit.
        :type model_id: str (uuid)
        :param model: The model object to commit
        :type model: BytesIO
        :param session_id: Unique identifier for the session
        :type session_id: str
        """
        helper = self.get_helper()
        if model is not None:
            outfile_name = helper.save(model)
            logger.info("Saving model file temporarily to {}".format(outfile_name))
            logger.info("CONTROL: Uploading model to object store...")
            model_id = self.repository.set_model(outfile_name, is_file=True)

            logger.info("CONTROL: Deleting temporary model file...")
            os.unlink(outfile_name)

        logger.info("Committing model {} to global model trail in statestore...".format(model_id))

        parent_model = None
        if session_id:
            last_model_of_session = self.db.model_store.list(1, 0, "committed_at", SortOrder.DESCENDING, session_id=session_id)
            if len(last_model_of_session) == 1:
                parent_model = last_model_of_session[0].model_id
            else:
                session = self.db.session_store.get(session_id)
                parent_model = session.seed_model_id

        new_model = ModelDTO()
        new_model.model_id = model_id
        new_model.parent_model = parent_model
        new_model.session_id = session_id
        new_model.name = name

        try:
            self.db.model_store.add(new_model)
        except Exception as e:
            logger.error("Failed to commit model to global model trail: {}".format(e))
            raise Exception("Failed to commit model to global model trail")

        return model_id

    def get_control_state(self):
        """Get the current state of the control.

        :return: The current state.
        :rtype: :class:`fedn.network.state.ReducerState`
        """
        return self.get_control().get_state()

    def get_number_of_available_clients(self, client_ids: list[str]):
        result = 0
        for combiner in self.get_combiners():
            try:
                active_clients = combiner.list_active_clients()
                if active_clients is not None:
                    if client_ids is not None:
                        filtered = [item for item in active_clients if item.client_id in client_ids]
                        result += len(filtered)
                    else:
                        result += len(active_clients)
            except CombinerUnavailableError:
                return 0
        return result

    def get_compute_package(self, compute_package=""):
        """:param compute_package:
        :return:
        """
        if compute_package == "":
            compute_package = self.get_compute_package_name()
        if compute_package:
            return self.repository.get_compute_package(compute_package)
        else:
            return None

    def get_compute_package_name(self):
        """:return:"""
        definition = self.db.package_store.get_active()
        if definition:
            try:
                package_name = definition.storage_file_name
                return package_name
            except (IndexError, KeyError):
                logger.error("No context filename set for compute context definition")
                return None
        else:
            return None
