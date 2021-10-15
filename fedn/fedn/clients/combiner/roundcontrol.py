import time
import json
import os
import queue
import tempfile
import time
import uuid
import sys
import queue

import fedn.common.net.grpc.fedn_pb2 as fedn
from threading import Thread, Lock
from fedn.utils.helpers import get_helper
 
class RoundControl:
    """ Combiner level round controller.  
   
    The controller recieves round configurations from the global controller  
    and acts on them by soliciting model updates and model validations
    from the connected clients.

    :param id: A reference to id of :class: `fedn.combiner.Combiner` 
    :type id: str
    :param storage: Model repository for :class: `fedn.combiner.Combiner` 
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.combiner.Combiner`
    :type server: class: `fedn.combiner.Combiner` 
    :param modelservice: A handle to the model service :class: `fedn.clients.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.clients.combiner.modelservice.ModelService`
    """

    def __init__(self, id, storage, server, modelservice):

        self.id = id
        self.round_configs = queue.Queue()
        self.storage = storage
        self.server = server
        self.modelservice = modelservice
        self.config = {}

        # TODO, make runtime configurable
        from fedn.aggregators.fedavg import FedAvgAggregator
        self.aggregator = FedAvgAggregator(self.id, self.storage, self.server, self.modelservice, self)

    def push_round_config(self, round_config):
        """ Recieve a round_config (job description) and push on the queue. 

        :param round_config: A dict containing round configurations.
        :type round_config: dict
        :return: A generated job id (universally unique identifier) for the round configuration 
        :rtype: str
        """
        try:
            import uuid
            round_config['_job_id'] = str(uuid.uuid4())
            self.round_configs.put(round_config)
        except:
            self.server.report_status("ROUNDCONTROL: Failed to push round config.", flush=True)
            raise
        return round_config['_job_id']
        
    def load_model_fault_tolerant(self, model_id, retry=3):
        """Load model update object.

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
        if model_str == None:
            model_str = self.modelservice.get_model(model_id)
            # TODO: use retrying library
            tries = 0
            while tries < retry:
                tries += 1
                if not model_str or sys.getsizeof(model_str) == 80:
                    self.server.report_status("ROUNDCONTROL: Model download failed. retrying", flush=True)
                    import time
                    time.sleep(1)
                    model_str = self.modelservice.get_model(model_id)

        return model_str

    def _training_round(self, config, clients):
        """Send model update requests to clients and aggregate results. 

        :param config: [description]
        :type config: [type]
        :param clients: [description]
        :type clients: [type]
        :return: [description]
        :rtype: [type]
        """

        # We flush the queue at a beginning of a round (no stragglers allowed)
        # TODO: Support other ways to handle stragglers. 
        with self.aggregator.model_updates.mutex:
            self.aggregator.model_updates.queue.clear()

        self.server.report_status("ROUNDCONTROL: Initiating training round, participating members: {}".format(clients))
        self.server.request_model_update(config['model_id'], clients=clients)

        meta = {}
        meta['nr_expected_updates'] = len(clients)
        meta['nr_required_updates'] = int(config['clients_required'])
        meta['timeout'] = float(config['round_timeout'])
        tic = time.time()
        model = None
        data = None
        try:
            helper = get_helper(config['helper_type'])
            model, data = self.aggregator.combine_models(nr_expected_models=len(clients),
                                              nr_required_models=int(config['clients_required']),
                                              helper=helper, timeout=float(config['round_timeout']))
        except Exception as e:
            print("TRAINING ROUND FAILED AT COMBINER! {}".format(e), flush=True)
        meta['time_combination'] = time.time() - tic
        meta['aggregation_time'] = data
        return model, meta

    def _validation_round(self, config, clients, model_id):
        """[summary]

        :param config: [description]
        :type config: [type]
        :param clients: [description]
        :type clients: [type]
        :param model_id: [description]
        :type model_id: [type]
        """
        self.server.request_model_validation(model_id, clients=clients)

    def stage_model(self, model_id, timeout_retry=3, retry=2):
        """Download model from persistent storage.

        :param model_id: ID of the model update object to stage. 
        :type model_id: str
        :param timeout_retry: Sleep before retrying download again (sec), defaults to 3
        :type timeout_retry: int, optional
        :param retry: Number of retries, defaults to 2
        :type retry: int, optional
        """

        # If the model is already in memory at the server we do not need to do anything.
        if self.modelservice.models.exist(model_id):
            return

        # If it is not there, download it from storage and stage it in memory at the server. 
        tries = 0
        while True:
            try:
                model = self.storage.get_model_stream(model_id)
                if model:
                    break
            except Exception as e:
                self.server.report_status("ROUNDCONTROL: Could not fetch model from storage backend, retrying.",
                                   flush=True)
                time.sleep(timeout_retry)
                tries += 1
                if tries > retry:
                    self.server.report_status("ROUNDCONTROL: Failed to stage model {} from storage backend!".format(model_id), flush=True)
                    return

        self.modelservice.set_model(model, model_id)
        
    def __assign_round_clients(self, n, type="trainers"):
        """ Obtain a list of clients (trainers or validators) to talk to in a round. 

        :param n: Size of a random set taken from active trainers (clients), if n > "active trainers" all is used
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
            self.server.report_status("ROUNDCONTROL(ERROR): {} is not a supported type of client".format(type), flush=True)
            raise


        # If the number of requested trainers exceeds the number of available, use all available. 
        if n > len(clients):
            n = len(clients)
            
        # If not, we pick a random subsample of all available clients.
        import random
        clients = random.sample(clients, n)

        return clients

    def __check_nr_round_clients(self, config, timeout=0.0):
        """Check that the minimal number of required clients to start a round are connected. 

        :param config: [description]
        :type config: [type]
        :param timeout: [description], defaults to 0.0
        :type timeout: float, optional
        :return: [description]
        :rtype: [type]
        """

        import time
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

    def execute_validation(self, round_config):
        """ Coordinate validation rounds as specified in config. 

        :param round_config: [description]
        :type round_config: [type]
        """
        model_id = round_config['model_id']
        self.server.report_status("COMBINER orchestrating validation of model {}".format(model_id))
        self.stage_model(model_id)
        validators = self.__assign_round_clients(self.server.max_clients,type="validators")
        self._validation_round(round_config,validators,model_id)        

    def execute_training(self, config):
        """ Coordinates clients to execute training and validation tasks. """

        round_meta = {}
        round_meta['config'] = config
        round_meta['round_id'] = config['round_id']

        self.stage_model(config['model_id'])

        # Execute the configured number of rounds
        round_meta['local_round'] = {}
        for r in range(1, int(config['rounds']) + 1):
            self.server.report_status("ROUNDCONTROL: Starting training round {}".format(r), flush=True)
            clients = self.__assign_round_clients(self.server.max_clients)
            model, meta = self._training_round(config, clients)
            round_meta['local_round'][str(r)] = meta
            if model is None:
                self.server.report_status("\t Failed to update global model in round {0}!".format(r))

        if model is not None:
            helper = get_helper(config['helper_type'])
            a = helper.serialize_model_to_BytesIO(model)
            # Send aggregated model to server 
            model_id = str(uuid.uuid4())
            self.modelservice.set_model(a, model_id)
            a.close()

            # Update Combiner latest model
            self.server.set_active_model(model_id)

            print("------------------------------------------")
            self.server.report_status("ROUNDCONTROL: TRAINING ROUND COMPLETED.", flush=True)
            print("\n")
        return round_meta

    def run(self):
        """ Main control loop. Sequentially execute rounds based on round config. 

        """
        try:
            while True:
                try:
                    round_config = self.round_configs.get(block=False)

                    ready = self.__check_nr_round_clients(round_config)
                    if ready:
                        if round_config['task'] == 'training':
                            tic = time.time()
                            round_meta = self.execute_training(round_config)
                            round_meta['time_exec_training'] = time.time() - tic
                            round_meta['name'] = self.id
                            self.server.tracer.set_round_meta(round_meta)
                        elif round_config['task'] == 'validation':
                            self.execute_validation(round_config)
                        else:
                            self.server.report_status("ROUNDCONTROL: Round config contains unkown task type.", flush=True)
                    else:
                        self.server.report_status("ROUNDCONTROL: Failed to meet client allocation requirements for this round config.", flush=True)

                except queue.Empty:
                    time.sleep(1)

        except (KeyboardInterrupt, SystemExit):
            pass
