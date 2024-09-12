import ast
import queue
import random
import sys
import time
import uuid
from typing import TypedDict

from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import get_aggregator
from fedn.network.combiner.modelservice import load_model_from_BytesIO, serialize_model_to_BytesIO
from fedn.network.combiner.shared import modelservice, repository
from fedn.utils.helpers.helpers import get_helper
from fedn.utils.parameters import Parameters


class RoundConfig(TypedDict):
    """Round configuration.

    :param _job_id: A universally unique identifier for the round. Set by Combiner.
    :type _job_id: str
    :param committed_at: The time the round was committed. Set by Controller.
    :type committed_at: str
    :param task: The task to perform in the round. Set by Controller. Supported tasks are "training", "validation", and "inference".
    :type task: str
    :param round_id: The round identifier as str(int)
    :type round_id: str
    :param round_timeout: The round timeout in seconds. Set by user interfaces or Controller.
    :type round_timeout: str
    :param rounds: The number of rounds. Set by user interfaces.
    :param model_id: The model identifier. Set by user interfaces or Controller (get_latest_model).
    :type model_id: str
    :param model_version: The model version. Currently not used.
    :type model_version: str
    :param model_type: The model type. Currently not used.
    :type model_type: str
    :param model_size: The size of the model. Currently not used.
    :type model_size: int
    :param model_parameters: The model parameters. Currently not used.
    :type model_parameters: dict
    :param model_metadata: The model metadata. Currently not used.
    :type model_metadata: dict
    :param session_id: The session identifier. Set by (Controller?).
    :type session_id: str
    :param helper_type: The helper type.
    :type helper_type: str
    :param aggregator: The aggregator type.
    :type aggregator: str
    """

    _job_id: str
    committed_at: str
    task: str
    round_id: str
    round_timeout: str
    rounds: int
    model_id: str
    model_version: str
    model_type: str
    model_size: int
    model_parameters: dict
    model_metadata: dict
    session_id: str
    helper_type: str
    aggregator: str


class ModelUpdateError(Exception):
    pass


class RoundHandler:
    """Round handler.

    The round handler processes requests from the global controller
    to produce model updates and perform model validations.

    :param aggregator_name: The name of the aggregator plugin module.
    :type aggregator_name: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    """

    def __init__(self, server):
        """Initialize the RoundHandler."""
        self.round_configs = queue.Queue()
        self.storage = repository
        self.server = server
        self.modelservice = modelservice

    def set_aggregator(self, aggregator):
        self.aggregator = get_aggregator(aggregator, self.storage, self.server, self.modelservice, self)

    def push_round_config(self, round_config: RoundConfig) -> str:
        """Add a round_config (job description) to the inbox.

        :param round_config: A dict containing the round configuration (from global controller).
        :type round_config: dict
        :return: A job id (universally unique identifier) for the round.
        :rtype: str
        """
        try:
            round_config["_job_id"] = str(uuid.uuid4())
            self.round_configs.put(round_config)
        except Exception:
            logger.error("Failed to push round config.")
            raise
        return round_config["_job_id"]

    def load_model_update(self, helper, model_id):
        """Load model update with id model_id into its memory representation.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`
        :param model_id: The ID of the model update, UUID in str format
        :type model_id: str
        """
        model_str = self.load_model_update_str(model_id)
        if model_str:
            try:
                model = load_model_from_BytesIO(model_str.getbuffer(), helper)
            except IOError:
                logger.warning("AGGREGATOR({}): Failed to load model!".format(self.name))
        else:
            raise ModelUpdateError("Failed to load model.")

        return model

    def load_model_update_str(self, model_id, retry=3):
        """Load model update object and return it as BytesIO.

        :param model_id: The ID of the model
        :type model_id: str
        :param retry: number of times retrying load model update, defaults to 3
        :type retry: int, optional
        :return: Updated model
        :rtype: class: `io.BytesIO`
        """
        # Try reading model update from local disk/combiner memory
        model_str = self.modelservice.temp_model_storage.get(model_id)
        # And if we cannot access that, try downloading from the server
        if model_str is None:
            model_str = self.modelservice.get_model(model_id)
            # TODO: use retrying library
            tries = 0
            while tries < retry:
                tries += 1
                if not model_str or sys.getsizeof(model_str) == 80:
                    logger.warning("Model download failed. retrying")
                    time.sleep(1)
                    model_str = self.modelservice.get_model(model_id)

        return model_str

    def waitforit(self, config, buffer_size=100, polling_interval=0.1):
        """Defines the policy for how long the server should wait before starting to aggregate models.

        The policy is as follows:
            1. Wait a maximum of time_window time until the round times out.
            2. Terminate if a preset number of model updates (buffer_size) are in the queue.

        :param config: The round config object
        :type config: dict
        :param buffer_size: The number of model updates to wait for before starting aggregation, defaults to 100
        :type buffer_size: int, optional
        :param polling_interval: The polling interval, defaults to 0.1
        :type polling_interval: float, optional
        """
        time_window = float(config["round_timeout"])

        tt = 0.0
        while tt < time_window:
            if self.aggregator.model_updates.qsize() >= buffer_size:
                break

            time.sleep(polling_interval)
            tt += polling_interval

    def _training_round(self, config, clients):
        """Send model update requests to clients and aggregate results.

        :param config: The round config object (passed to the client).
        :type config: dict
        :param clients: clients to participate in the training round
        :type clients: list
        :return: an aggregated model and associated metadata
        :rtype: model, dict
        """
        logger.info("ROUNDHANDLER: Initiating training round, participating clients: {}".format(clients))

        meta = {}
        meta["nr_expected_updates"] = len(clients)
        meta["nr_required_updates"] = int(config["clients_required"])
        meta["timeout"] = float(config["round_timeout"])

        session_id = config["session_id"]
        model_id = config["model_id"]

        # Request model updates from all active clients.
        self.server.request_model_update(session_id=session_id, model_id=model_id, config=config, clients=clients)

        # If buffer_size is -1 (default), the round terminates when/if all clients have completed.
        if int(config["buffer_size"]) == -1:
            buffer_size = len(clients)
        else:
            buffer_size = int(config["buffer_size"])

        # Wait / block until the round termination policy has been met.
        self.waitforit(config, buffer_size=buffer_size)

        tic = time.time()
        model = None
        data = None

        try:
            helper = get_helper(config["helper_type"])
            logger.info("Config delete_models_storage: {}".format(config["delete_models_storage"]))
            if config["delete_models_storage"] == "True":
                delete_models = True
            else:
                delete_models = False

            if "aggregator_kwargs" in config.keys():
                dict_parameters = ast.literal_eval(config["aggregator_kwargs"])
                parameters = Parameters(dict_parameters)
            else:
                parameters = None

            model, data = self.aggregator.combine_models(helper=helper, delete_models=delete_models, parameters=parameters)
        except Exception as e:
            logger.warning("AGGREGATION FAILED AT COMBINER! {}".format(e))

        meta["time_combination"] = time.time() - tic
        meta["aggregation_time"] = data
        return model, meta

    def _validation_round(self, session_id, model_id, clients):
        """Send model validation requests to clients.

        :param config: The round config object (passed to the client).
        :type config: dict
        :param clients: clients to send validation requests to
        :type clients: list
        :param model_id: The ID of the model to validate
        :type model_id: str
        """
        self.server.request_model_validation(session_id, model_id, clients=clients)

    def _inference_round(self, session_id: str, model_id: str, clients: list):
        """Send model inference requests to clients.

        :param config: The round config object (passed to the client).
        :type config: dict
        :param clients: clients to send inference requests to
        :type clients: list
        :param model_id: The ID of the model to use for inference
        :type model_id: str
        """
        self.server.request_model_inference(session_id, model_id, clients=clients)

    def stage_model(self, model_id, timeout_retry=3, retry=2):
        """Download a model from persistent storage and set in modelservice.

        :param model_id: ID of the model update object to stage.
        :type model_id: str
        :param timeout_retry: Sleep before retrying download again(sec), defaults to 3
        :type timeout_retry: int, optional
        :param retry: Number of retries, defaults to 2
        :type retry: int, optional
        """
        # If the model is already in memory at the server we do not need to do anything.
        if self.modelservice.temp_model_storage.exist(model_id):
            logger.info("Model already exists in memory, skipping model staging.")
            return
        logger.info("Model Staging, fetching model from storage...")
        # If not, download it and stage it in memory at the combiner.
        tries = 0
        while True:
            try:
                model = self.storage.get_model_stream(model_id)
                if model:
                    break
            except Exception:
                logger.warning("Could not fetch model from storage backend, retrying.")
                time.sleep(timeout_retry)
                tries += 1
                if tries > retry:
                    logger.error("Failed to stage model {} from storage backend!".format(model_id))
                    raise

        self.modelservice.set_model(model, model_id)

    def _assign_round_clients(self, n, type="trainers"):
        """Obtain a list of clients(trainers or validators) to ask for updates in this round.

        :param n: Size of a random set taken from active trainers(clients), if n > "active trainers" all is used
        :type n: int
        :param type: type of clients, either "trainers" or "validators", defaults to "trainers"
        :type type: str, optional
        :return: Set of clients
        :rtype: list
        """
        if type == "validators":
            clients = self.server.get_active_validators()
        elif type == "trainers":
            clients = self.server.get_active_trainers()
        else:
            logger.error("(ERROR): {} is not a supported type of client".format(type))

        # If the number of requested trainers exceeds the number of available, use all available.
        n = min(n, len(clients))

        # If not, we pick a random subsample of all available clients.
        clients = random.sample(clients, n)

        return clients

    def _check_nr_round_clients(self, config):
        """Check that the minimal number of clients required to start a round are available.

        :param config: The round config object.
        :type config: dict
        :param timeout: Timeout in seconds, defaults to 0.0
        :type timeout: float, optional
        :return: True if the required number of clients are available, False otherwise.
        :rtype: bool
        """
        active = self.server.nr_active_trainers()
        if active >= int(config["clients_required"]):
            logger.info("Number of clients required ({0}) to start round met {1}.".format(config["clients_required"], active))
            return True
        else:
            logger.info("Too few clients to start round.")
            return False

    def execute_validation_round(self, session_id, model_id):
        """Coordinate validation rounds as specified in config.

        :param round_config: The round config object.
        :type round_config: dict
        """
        logger.info("COMBINER orchestrating validation of model {}".format(model_id))
        self.stage_model(model_id)
        validators = self._assign_round_clients(self.server.max_clients, type="validators")
        self._validation_round(session_id, model_id, validators)

    def execute_inference_round(self, session_id: str, model_id: str) -> None:
        """Coordinate inference rounds as specified in config.

        :param round_config: The round config object.
        :type round_config: dict
        """
        logger.info("COMBINER orchestrating inference using model {}".format(model_id))
        self.stage_model(model_id)
        # TODO: Implement inference client type
        clients = self._assign_round_clients(self.server.max_clients, type="validators")
        self._inference_round(session_id, model_id, clients)

    def execute_training_round(self, config):
        """Coordinates clients to execute training tasks.

        :param config: The round config object.
        :type config: dict
        :return: metadata about the training round.
        :rtype: dict
        """
        logger.info("Processing training round,  job_id {}".format(config["_job_id"]))

        data = {}
        data["config"] = config
        data["round_id"] = config["round_id"]

        # Download model to update and set in temp storage.
        self.stage_model(config["model_id"])

        clients = self._assign_round_clients(self.server.max_clients)
        model, meta = self._training_round(config, clients)
        data["data"] = meta

        if model is None:
            logger.warning("\t Failed to update global model in round {0}!".format(config["round_id"]))

        if model is not None:
            helper = get_helper(config["helper_type"])
            a = serialize_model_to_BytesIO(model, helper)
            model_id = self.storage.set_model(a.read(), is_file=False)
            a.close()
            data["model_id"] = model_id

            logger.info("TRAINING ROUND COMPLETED. Aggregated model id: {}, Job id: {}".format(model_id, config["_job_id"]))

        # Delete temp model
        self.modelservice.temp_model_storage.delete(config["model_id"])
        return data

    def run(self, polling_interval=1.0):
        """Main control loop. Execute rounds based on round config on the queue.

        :param polling_interval: The polling interval in seconds for checking if a new job/config is available.
        :type polling_interval: float
        """
        try:
            while True:
                try:
                    round_config = self.round_configs.get(block=False)
                    session_id = round_config["session_id"]
                    model_id = round_config["model_id"]

                    # Check that the minimum allowed number of clients are connected
                    ready = self._check_nr_round_clients(round_config)
                    round_meta = {}

                    if ready:
                        if round_config["task"] == "training":
                            tic = time.time()
                            round_meta = self.execute_training_round(round_config)
                            round_meta["time_exec_training"] = time.time() - tic
                            round_meta["status"] = "Success"
                            round_meta["name"] = self.server.id
                            self.server.statestore.set_round_combiner_data(round_meta)
                        elif round_config["task"] == "validation":
                            self.execute_validation_round(session_id, model_id)
                        elif round_config["task"] == "inference":
                            self.execute_inference_round(session_id, model_id)
                        else:
                            logger.warning("config contains unkown task type.")
                    else:
                        round_meta = {}
                        round_meta["status"] = "Failed"
                        round_meta["reason"] = "Failed to meet client allocation requirements for this round config."
                        logger.warning("{0}".format(round_meta["reason"]))

                    self.round_configs.task_done()
                except queue.Empty:
                    time.sleep(polling_interval)

        except (KeyboardInterrupt, SystemExit):
            pass
