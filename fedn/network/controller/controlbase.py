from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import time

from fedn.common.log_config import logger
import fedn.network.grpc.fedn_pb2 as fedn_proto
from fedn.network.combiner.roundhandler import RoundConfig
from fedn.network.common.interfaces import CombinerInterface, CombinerUnavailableError
from fedn.network.common.network import Network
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.stores.dto.round import RoundDTO


class ControlBase(ABC):
    """Base class and interface for a global controller.
        Override this class to implement a global training strategy (control).

    :param statestore: The statestore object.
    :type statestore: :class:`fedn.network.statestore.statestorebase.StateStoreBase`
    """

    repository: Repository

    @abstractmethod
    def __init__(
        self,
        network_id: str,
        repository: Repository,
        db: DatabaseConnection,
    ):
        """Constructor."""
        self.network = Network(db, repository)

        self.repository = repository
        self.db = db
        self._active_clients_cache = {} 
        self.COMBINER_CACHE_COOLDOWN = 180.0  # seconds


    @abstractmethod
    def round(self, config, round_number):
        pass

    @abstractmethod
    def reduce(self, combiners):
        pass

    def get_latest_round_id(self) -> int:
        return self.db.round_store.get_latest_round_id()

    def set_compute_package(self, filename, path):
        """Persist the configuration for the compute package."""
        self.repository.set_compute_package(filename, path)

    def set_session_status(self, session_id: str, status: str) -> Tuple[bool, Any]:
        """Set the round round stats.

        :param round_id: The round unique identifier
        :type round_id: str
        :param status: The status
        :type status: str
        """
        session = self.db.session_store.get(session_id)
        session.status = status
        self.db.session_store.update(session)

    def get_session_status(self, session_id: str):
        """Get the status of a session.

        :param session_id: The session unique identifier
        :type session_id: str
        :return: The status
        :rtype: str
        """
        session = self.db.session_store.get(session_id)
        return session.status

    def set_session_config(self, session_id: str, config: dict) -> Tuple[bool, Any]:
        """Set the model id for a session

        :param session_id: The session unique identifier
        :type session_id: str
        :param config: The session config
        :type config: dict
        """
        session = self.db.session_store.get(session_id)
        session.session_config.patch_with(config)

        self.db.session_store.update(session)

    def create_round(self, round_data):
        """Initialize a new round in backend db."""
        round = RoundDTO(**round_data)
        self.db.round_store.add(round)

    def set_round_data(self, round_id: str, round_data: dict):
        """Set round data.

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_data: The status
        :type status: dict
        """
        round = self.db.round_store.get(round_id)
        round.round_data = round_data
        self.db.round_store.update(round)

    def set_round_status(self, round_id: str, status: str):
        """Set the round round stats.

        :param round_id: The round unique identifier
        :type round_id: str
        :param status: The status
        :type status: str
        """
        round = self.db.round_store.get(round_id)
        round.status = status
        self.db.round_store.update(round)

    def set_round_config(self, round_id: str, round_config: RoundConfig):
        """Upate round in backend db.

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_config: The round configuration
        :type round_config: dict
        """
        round = self.db.round_store.get(round_id)
        round.round_config = round_config
        self.db.round_store.update(round)

    def request_model_updates(self, combiners: List[Tuple[CombinerInterface, Dict]]):
        """Ask Combiner server to produce a model update.

        :param combiners: A list of combiners
        :type combiners: tuple (combiner, combiner_round_config)
        """
        cl = []
        participating_combiners = []
        for combiner, combiner_round_config in combiners:
            try:
                response = combiner.submit(fedn_proto.Command.START, combiner_round_config)
                cl.append((combiner, response))
                participating_combiners.append((combiner, combiner_round_config))
            except CombinerUnavailableError:
                self._handle_unavailable_combiner(combiner)
                continue
        return cl, participating_combiners

    def get_combiner(self, name):
        for combiner in self.network.get_combiners():
            if combiner.name == name:
                return combiner
        return None

    def get_participating_combiners(self, combiner_round_config) -> List[Tuple[CombinerInterface, Dict]]:
        """Assemble a list of combiners able to participate in a round
           according to combiner_round_config.
        """
        combiners = []
        for combiner in self.network.get_combiners():
            nr_active_clients = self._get_nr_active_clients_throttled(combiner)
            if nr_active_clients is None:
                logger.warning(f"Combiner {combiner.name} returned nr_active_clients: None.")
                continue

            clients_required = int(combiner_round_config["clients_required"])
            is_participating = self.evaluate_round_participation_policy(clients_required, nr_active_clients)
            if is_participating:
                combiners.append((combiner, combiner_round_config))

        return combiners
    
    def _get_nr_active_clients_throttled(self, combiner: CombinerInterface) -> int:
        """Return the number of active clients for a combiner, but avoid
           calling `list_active_clients()` if we recently did so.
        """
        now = time.time()
        cache_entry = self._active_clients_cache.get(combiner.name)

        # if we have a cache entry and the time between last call is less than COMBINER_CACHE_COOLDOWN, reuse it.
        if cache_entry:
            last_timestamp, nr_active = cache_entry
            if (now - last_timestamp) < self.COMBINER_CACHE_COOLDOWN:
                return nr_active

        # otherwise, do a fresh call
        try:
            nr_active_clients = len(combiner.list_active_clients())
        except CombinerUnavailableError:
            logger.warning(f"Combiner {combiner.name} is unavailable.")
            return None

        # update the cache
        self._active_clients_cache[combiner.name] = (time.time(), nr_active_clients)
        return nr_active_clients
    

    def _handle_unavailable_combiner(self, combiner):
        logger.warning(f"Ignoring unavailable combiner {combiner.name}.")

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
