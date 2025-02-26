import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Tuple

import fedn.utils.helpers.helpers
from fedn.common.log_config import logger
from fedn.network.api.network import Network
from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.combiner.roundhandler import RoundConfig
from fedn.network.state import ReducerState
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.stores.client_store import ClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore
from fedn.network.storage.statestore.stores.model_store import ModelStore
from fedn.network.storage.statestore.stores.package_store import PackageStore
from fedn.network.storage.statestore.stores.round_store import RoundStore
from fedn.network.storage.statestore.stores.session_store import MongoDBSessionStore

# Maximum number of tries to connect to statestore and retrieve storage configuration
MAX_TRIES_BACKEND = os.getenv("MAX_TRIES_BACKEND", 10)


class UnsupportedStorageBackend(Exception):
    pass


class MisconfiguredStorageBackend(Exception):
    pass


class MisconfiguredHelper(Exception):
    pass


class ControlBase(ABC):
    """Base class and interface for a global controller.
        Override this class to implement a global training strategy (control).

    :param statestore: The statestore object.
    :type statestore: :class:`fedn.network.statestore.statestorebase.StateStoreBase`
    """

    @abstractmethod
    def __init__(
        self,
        network_id: str,
        session_store: MongoDBSessionStore,
        model_store: ModelStore,
        round_store: RoundStore,
        package_store: PackageStore,
        combiner_store: CombinerStore,
        client_store: ClientStore,
        model_repository: Repository,
    ):
        """Constructor."""
        self._state = ReducerState.setup

        self.session_store = session_store
        self.model_store = model_store
        self.round_store = round_store
        self.package_store = package_store
        self.network = Network(self, network_id, combiner_store, client_store)

        self.model_repository = model_repository

        self._state = ReducerState.idle

    @abstractmethod
    def round(self, config, round_number):
        pass

    @abstractmethod
    def reduce(self, combiners):
        pass

    def get_helper(self):
        """Get a helper instance from global config.

        :return: Helper instance.
        :rtype: :class:`fedn.utils.plugins.helperbase.HelperBase`
        """
        helper_type: str = None

        try:
            active_package = self.package_store.get_active()
            helper_type = active_package["helper"]
        except Exception:
            logger.error("Failed to get active helper")

        helper = fedn.utils.helpers.helpers.get_helper(helper_type)
        if not helper:
            raise MisconfiguredHelper("Unsupported helper type {}, please configure compute_package.helper !".format(helper_type))
        return helper

    def get_state(self):
        """Get the current state of the controller.

        :return: The current state.
        :rtype: :class:`fedn.network.state.ReducerState`
        """
        return self._state

    def idle(self):
        """Check if the controller is idle.

        :return: True if idle, False otherwise.
        :rtype: bool
        """
        if self._state == ReducerState.idle:
            return True
        else:
            return False

    def get_latest_round_id(self) -> int:
        return self.round_store.get_latest_round_id()

    def get_compute_package_name(self):
        """:return:"""
        definition = self.package_store.get_active()
        if definition:
            try:
                package_name = definition["storage_file_name"]
                return package_name
            except (IndexError, KeyError):
                logger.error("No context filename set for compute context definition")
                return None
        else:
            return None

    def set_compute_package(self, filename, path):
        """Persist the configuration for the compute package."""
        self.model_repository.set_compute_package(filename, path)

    def get_compute_package(self, compute_package=""):
        """:param compute_package:
        :return:
        """
        if compute_package == "":
            compute_package = self.get_compute_package_name()
        if compute_package:
            return self.model_repository.get_compute_package(compute_package)
        else:
            return None

    def set_session_status(self, session_id: str, status: str) -> Tuple[bool, Any]:
        """Set the round round stats.

        :param round_id: The round unique identifier
        :type round_id: str
        :param status: The status
        :type status: str
        """
        session = self.session_store.get(session_id)
        session["status"] = status
        updated, msg = self.session_store.update(session["id"], session)
        if not updated:
            raise Exception(msg)

    def get_session_status(self, session_id: str):
        """Get the status of a session.

        :param session_id: The session unique identifier
        :type session_id: str
        :return: The status
        :rtype: str
        """
        session = self.session_store.get(session_id)
        return session["status"]

    def set_session_config(self, session_id: str, config: dict) -> Tuple[bool, Any]:
        """Set the model id for a session.

        :param session_id: The session unique identifier
        :type session_id: str
        :param config: The session config
        :type config: dict
        """
        session = self.session_store.get(session_id)
        session["session_config"] = config
        updated, msg = self.session_store.update(session["id"], session)
        if not updated:
            raise Exception(msg)

    def create_round(self, round_data):
        """Initialize a new round in backend db."""
        self.round_store.add(round_data)

    def set_round_data(self, round_id: str, round_data: dict):
        """Set round data.

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_data: The status
        :type status: dict
        """
        round = self.round_store.get(round_id)
        round["round_data"] = round_data
        updated, _ = self.round_store.update(round["id"], round)
        if not updated:
            raise Exception("Failed to update round")

    def set_round_status(self, round_id: str, status: str):
        """Set the round round stats.

        :param round_id: The round unique identifier
        :type round_id: str
        :param status: The status
        :type status: str
        """
        round = self.round_store.get(round_id)
        round["status"] = status
        updated, _ = self.round_store.update(round["id"], round)
        if not updated:
            raise Exception("Failed to update round")

    def set_round_config(self, round_id: str, round_config: RoundConfig):
        """Upate round in backend db.

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_config: The round configuration
        :type round_config: dict
        """
        round = self.round_store.get(round_id)
        round["round_config"] = round_config
        updated, _ = self.round_store.update(round["id"], round)
        if not updated:
            raise Exception("Failed to update round")

    def request_model_updates(self, combiners):
        """Ask Combiner server to produce a model update.

        :param combiners: A list of combiners
        :type combiners: tuple (combiner, combiner_round_config)
        """
        cl = []
        for combiner, combiner_round_config in combiners:
            response = combiner.submit(combiner_round_config)
            cl.append((combiner, response))
        return cl

    def commit(self, model_id: str, model: dict = None, session_id: str = None, name: str = None):
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
            logger.info("CONTROL: Uploading model to Minio...")
            model_id = self.model_repository.set_model(outfile_name, is_file=True)

            logger.info("CONTROL: Deleting temporary model file...")
            os.unlink(outfile_name)

        logger.info("Committing model {} to global model trail in statestore...".format(model_id))

        active_model: str = None

        try:
            active_model = self.model_store.get_active()
        except Exception:
            logger.info("No active model, adding...")

        parent_model = None
        if active_model and session_id:
            parent_model = active_model

        committed_at = datetime.now()

        updated, _ = self.model_store.add(
            {
                "key": "models",
                "model": model_id,
                "parent_model": parent_model,
                "session_id": session_id,
                "committed_at": committed_at,
                "name": name,
            }
        )

        if not updated:
            raise Exception("Failed to commit model to global model trail")

        self.model_store.set_active(model_id)

    def get_combiner(self, name):
        for combiner in self.network.get_combiners():
            if combiner.name == name:
                return combiner
        return None

    def get_participating_combiners(self, combiner_round_config):
        """Assemble a list of combiners able to participate in a round as
        descibed by combiner_round_config.
        """
        combiners = []
        for combiner in self.network.get_combiners():
            try:
                # Current gRPC endpoint only returns active clients (both trainers and validators)
                nr_active_clients = len(combiner.list_active_clients())
            except CombinerUnavailableError:
                self._handle_unavailable_combiner(combiner)
                continue

            clients_required = int(combiner_round_config["clients_required"])
            is_participating = self.evaluate_round_participation_policy(clients_required, nr_active_clients)
            if is_participating:
                combiners.append((combiner, combiner_round_config))
        return combiners

    def evaluate_round_participation_policy(self, clients_required: int, nr_active_clients: int) -> bool:
        """Evaluate policy for combiner round-participation.
        A combiner participates if it is responsive and reports enough
        active clients to participate in the round.
        """
        if clients_required <= nr_active_clients:
            return True
        else:
            return False

    def evaluate_round_start_policy(self, combiners: list):
        """Check if the policy to start a round is met.

        :param combiners: A list of combiners
        :type combiners: list
        :return: True if the round policy is mer, otherwise False
        :rtype: bool
        """
        if len(combiners) > 0:
            return True
        else:
            return False

    def evaluate_round_validity_policy(self, round):
        """Check if the round is valid.

        At the end of the round, before committing a model to the global model trail,
        we check if the round validity policy has been met. This can involve
        e.g. asserting that a certain number of combiners have reported in an
        updated model, or that criteria on model performance have been met.

        :param round: The round object
        :rtype round: dict
        :return: True if the policy is met, otherwise False
        :rtype: bool
        """
        model_ids = []
        for combiner in round["combiners"]:
            try:
                model_ids.append(combiner["model_id"])
            except KeyError:
                pass

        if len(model_ids) == 0:
            return False

        return True

    def state(self):
        """Get the current state of the controller.

        :return: The state
        :rype: str
        """
        return self._state
