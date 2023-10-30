import queue
import random
import sys
import time
import uuid

from fedn.network.combiner.aggregators.aggregatorbase import get_aggregator
from fedn.utils.helpers import get_helper


class ModelUpdateError(Exception):
    pass


class RoundController:
    """ Round controller.

    The round controller recieves round configurations from the global controller
    and coordinates model updates and aggregation, and model validations.

    :param aggregator_name: The name of the aggregator plugin module.
    :type aggregator_name: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    """

    def __init__(self, aggregator_name, storage, server, modelservice):

        self.round_configs = queue.Queue()
        self.storage = storage
        self.server = server
        self.modelservice = modelservice
        self.aggregator = get_aggregator(aggregator_name, self.storage, self.server, self.modelservice, self)

    def push_round_config(self, round_config):
        """Add a round_config (job description) to the inbox.

        :param round_config: A dict containing the round configuration (from global controller).
        :type round_config: dict
        :return: A job id (universally unique identifier) for the round.
        :rtype: str
        """
        try:
            round_config['_job_id'] = str(uuid.uuid4())
            self.round_configs.put(round_config)
        except Exception:
            self.server.report_status(
                "ROUNDCONTROL: Failed to push round config.", flush=True)
            raise
        return round_config['_job_id']

    def load_model_update(self, helper, model_id):
        """Load model update in its native format.

        :param helper: An instance of :class: `fedn.utils.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.HelperBase`
        :param model_id: The ID of the model update, UUID in str format
        :type model_id: str
        """

        model_str = self.load_model_update_str(model_id)
        if model_str:
            try:
                model = self.modelservice.load_model_from_BytesIO(model_str.getbuffer(), helper)
            except IOError:
                self.server.report_status(
                    "AGGREGATOR({}): Failed to load model!".format(self.name))
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
        model_str = self.modelservice.models.get(model_id)
        # And if we cannot access that, try downloading from the server
        if model_str is None:
            model_str = self.modelservice.get_model(model_id)
            # TODO: use retrying library
            tries = 0
            while tries < retry:
                tries += 1
                if not model_str or sys.getsizeof(model_str) == 80:
                    self.server.report_status(
                        "ROUNDCONTROL: Model download failed. retrying", flush=True)

                    time.sleep(1)
                    model_str = self.modelservice.get_model(model_id)

        return model_str

    def waitforit(self, config, buffer_size=100, polling_interval=0.1):
        """ Defines the policy for how long the server should wait before starting to aggregate models.

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

        time_window = float(config['round_timeout'])

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

        self.server.report_status(
            "ROUNDCONTROL: Initiating training round, participating clients: {}".format(clients))

        meta = {}
        meta['nr_expected_updates'] = len(clients)
        meta['nr_required_updates'] = int(config['clients_required'])
        meta['timeout'] = float(config['round_timeout'])

        # Request model updates from all active clients.
        self.server.request_model_update(config, clients=clients)

        # If buffer_size is -1 (default), the round terminates when/if all clients have completed.
        if int(config['buffer_size']) == -1:
            buffer_size = len(clients)
        else:
            buffer_size = int(config['buffer_size'])

        # Wait / block until the round termination policy has been met.
        self.waitforit(config, buffer_size=buffer_size)

        tic = time.time()
        model = None
        data = None

        try:
            helper = get_helper(config['helper_type'])
            print("ROUNDCONTROL: Config delete_models_storage: {}".format(config['delete_models_storage']), flush=True)
            if config['delete_models_storage'] == 'True':
                delete_models = True
            else:
                delete_models = False
            model, data = self.aggregator.combine_models(helper=helper,
                                                         delete_models=delete_models)
        except Exception as e:
            print("AGGREGATION FAILED AT COMBINER! {}".format(e), flush=True)

        meta['time_combination'] = time.time() - tic
        meta['aggregation_time'] = data
        return model, meta

    def _validation_round(self, config, clients, model_id):
        """Send model validation requests to clients.

        :param config: The round config object (passed to the client).
        :type config: dict
        :param clients: clients to send validation requests to
        :type clients: list
        :param model_id: The ID of the model to validate
        :type model_id: str
        """
        self.server.request_model_validation(model_id, config, clients)

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
        if self.modelservice.models.exist(model_id):
            print("MODEL EXISTST (NOT)", flush=True)
            return
        print("MODEL STAGING", flush=True)
        # If not, download it and stage it in memory at the combiner.
        tries = 0
        while True:
            try:
                model = self.storage.get_model_stream(model_id)
                if model:
                    break
            except Exception:
                self.server.report_status("ROUNDCONTROL: Could not fetch model from storage backend, retrying.",
                                          flush=True)
                time.sleep(timeout_retry)
                tries += 1
                if tries > retry:
                    self.server.report_status(
                        "ROUNDCONTROL: Failed to stage model {} from storage backend!".format(model_id), flush=True)
                    return

        self.modelservice.set_model(model, model_id)

    def _assign_round_clients(self, n, type="trainers"):
        """ Obtain a list of clients(trainers or validators) to ask for updates in this round.

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
            self.server.report_status(
                "ROUNDCONTROL(ERROR): {} is not a supported type of client".format(type), flush=True)
            raise

        # If the number of requested trainers exceeds the number of available, use all available.
        if n > len(clients):
            n = len(clients)

        # If not, we pick a random subsample of all available clients.
        clients = random.sample(clients, n)

        return clients

    def _check_nr_round_clients(self, config, timeout=0.0):
        """Check that the minimal number of clients required to start a round are available.

        :param config: The round config object.
        :type config: dict
        :param timeout: Timeout in seconds, defaults to 0.0
        :type timeout: float, optional
        :return: True if the required number of clients are available, False otherwise.
        :rtype: bool
        """

        ready = False
        t = 0.0
        while not ready:
            active = self.server.nr_active_trainers()

            if active >= int(config['clients_requested']):
                return True
            else:
                self.server.report_status("waiting for {} clients to get started, currently: {}".format(
                    int(config['clients_requested']) - active,
                    active), flush=True)
            if t >= timeout:
                if active >= int(config['clients_required']):
                    return True
                else:
                    return False

            time.sleep(1.0)
            t += 1.0

        return ready

    def execute_validation_round(self, round_config):
        """ Coordinate validation rounds as specified in config.

        :param round_config: The round config object.
        :type round_config: dict
        """
        model_id = round_config['model_id']
        self.server.report_status(
            "COMBINER orchestrating validation of model {}".format(model_id))
        self.stage_model(model_id)
        validators = self._assign_round_clients(
            self.server.max_clients, type="validators")
        self._validation_round(round_config, validators, model_id)

    def execute_training_round(self, config):
        """ Coordinates clients to execute training tasks.

        :param config: The round config object.
        :type config: dict
        :return: metadata about the training round.
        :rtype: dict
        """

        self.server.report_status(
            "ROUNDCONTROL: Processing training round,  job_id {}".format(config['_job_id']), flush=True)

        data = {}
        data['config'] = config
        data['round_id'] = config['round_id']

        # Make sure the model to update is available on this combiner.
        self.stage_model(config['model_id'])

        clients = self._assign_round_clients(self.server.max_clients)
        model, meta = self._training_round(config, clients)
        data['data'] = meta

        if model is None:
            self.server.report_status(
                "\t Failed to update global model in round {0}!".format(config['round_id']))

        if model is not None:
            helper = get_helper(config['helper_type'])
            a = self.modelservice.serialize_model_to_BytesIO(model, helper)
            # Send aggregated model to server
            model_id = str(uuid.uuid4())
            self.modelservice.set_model(a, model_id)
            a.close()
            data['model_id'] = model_id

            self.server.report_status(
                "ROUNDCONTROL: TRAINING ROUND COMPLETED. Aggregated model id: {}, Job id: {}".format(model_id, config['_job_id']), flush=True)

        return data

    def run(self, polling_interval=1.0):
        """ Main control loop. Execute rounds based on round config on the queue.

        :param polling_interval: The polling interval in seconds for checking if a new job/config is available.
        :type polling_interval: float
        """
        try:
            while True:
                try:
                    round_config = self.round_configs.get(block=False)

                    # Check that the minimum allowed number of clients are connected
                    ready = self._check_nr_round_clients(round_config)
                    round_meta = {}

                    if ready:
                        if round_config['task'] == 'training':
                            tic = time.time()
                            round_meta = self.execute_training_round(round_config)
                            round_meta['time_exec_training'] = time.time() - \
                                tic
                            round_meta['status'] = "Success"
                            round_meta['name'] = self.server.id
                            self.server.tracer.set_round_combiner_data(round_meta)
                        elif round_config['task'] == 'validation' or round_config['task'] == 'inference':
                            self.execute_validation_round(round_config)
                        else:
                            self.server.report_status(
                                "ROUNDCONTROL: Round config contains unkown task type.", flush=True)
                    else:
                        round_meta = {}
                        round_meta['status'] = "Failed"
                        round_meta['reason'] = "Failed to meet client allocation requirements for this round config."
                        self.server.report_status(
                            "ROUNDCONTROL: {0}".format(round_meta['reason']), flush=True)

                    self.round_configs.task_done()
                except queue.Empty:
                    time.sleep(polling_interval)

        except (KeyboardInterrupt, SystemExit):
            pass
