import json
import os
import queue
import tempfile
import time
import uuid
import sys

import fedn.common.net.grpc.fedn_pb2 as fedn
from threading import Thread, Lock
from  fedn.utils.helpers import get_helper 

class FEDAVGCombiner:
    """ 
        A Local SGD / Federated Averaging (FedAvg) aggregator. Coordinating the update of the Combiner global 
        model by requesting and aggregating model updates from Clients. 

    """

    def __init__(self, id, storage, server, modelservice):

        self.run_configs_lock = Lock()
        self.run_configs = []
        self.storage = storage
        self.id = id
        self.server = server 
        self.modelservice = modelservice

        self.config = {}
        self.validations = {}

        self.model_updates = queue.Queue()

    def report_status(self, msg, log_level=fedn.Status.INFO, type=None, request=None, flush=True):
        print("COMBINER({}):{} {}".format(self.id, log_level, msg), flush=flush)

    def receive_model_candidate(self, model_id):
        """ Callback when a new model version is reported by a client. """
        try:
            self.report_status("COMBINER: callback received model {}".format(model_id),
                               log_level=fedn.Status.INFO)
            # TODO - here would be a place to do some additional validation of the model contribution. 
            self.model_updates.put(model_id)
        except Exception as e:
            self.report_status("COMBINER: Failed to receive candidate model! {}".format(e),
                               log_level=fedn.Status.WARNING)
            self.report_status("Failed to receive candidate model!")
            pass

    def receive_validation(self, validation):
        """ Callback for a validation request """

        model_id = validation.model_id
        data = json.loads(validation.data)
        try:
            self.validations[model_id].append(data)
        except KeyError:
            self.validations[model_id] = [data]

        self.report_status("COMBINER: callback processed validation {}".format(validation.model_id),
                           log_level=fedn.Status.INFO)

    def _load_model_fault_tolerant(self,model_id):
        # Try reading it from local disk/combiner memory
        model_str = self.modelservice.models.get(model_id)
        # And if we cannot access that, try downloading from the server
        if model_str == None:
            model_str = self.modelservice.get_model(model_id)
            # TODO: use retrying library
            tries = 0
            while tries < 3:
                tries += 1
                if not model_str or sys.getsizeof(model_str) == 80:
                    self.report_status("COMBINER: Model download failed. retrying", flush=True)
                    import time
                    time.sleep(1)
                    model_str = self.modelservice.get_model(model_id)

        return model_str
 

    def combine_models(self, nr_expected_models=None, nr_required_models=1, timeout=180):
        """ Compute an iterative/running average of models arriving to the combiner. """

        import time
        round_time = 0.0
        print("COMBINER: combining model updates from Clients...")
        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0 

        polling_interval = 1.0
        nr_processed_models = 0
        while nr_processed_models < nr_expected_models:
            try:
                model_id = self.model_updates.get(block=False)
                self.report_status("Received model update with id {}".format(model_id))

                # Load the model update
                tic = time.time()
                model_str = self._load_model_fault_tolerant(model_id)
                if model_str:
                    try:
                        model_next = self.helper.load_model_from_BytesIO(model_str.getbuffer())
                    except IOError:
                        self.report_status("COMBINER: Failed to load model!")
                else: 
                    raise
    
                data['time_model_load'] += time.time()-tic

                # Aggregate
                tic = time.time()
                if nr_processed_models == 0:
                    model = model_next
                else:
                    model = self.helper.increment_average(model, model_next, nr_processed_models+1)                
                data['time_model_aggregation'] += time.time()-tic 

                nr_processed_models += 1
                self.model_updates.task_done()
            except queue.Empty:
                self.report_status("COMBINER: waiting for model updates: {} of {} completed.".format(nr_processed_models
                    ,nr_expected_models))
                time.sleep(polling_interval)
                round_time += polling_interval
            except IOError:
                self.report_status("COMBINER: Failed to read model update, skipping!")
                nr_expected_models -= 1
                if nr_expected_models <= 0:
                    return None
                self.model_updates.task_done()
            except Exception as e:
                self.report_status("COMBINER: Unknown exception in combine_models: {}".format(e))
                nr_expected_models -= 1
                if nr_expected_models <= 0:
                    return None
                self.model_updates.task_done()

            if round_time >= timeout:
                self.report_status("COMBINER: training round timed out.", log_level=fedn.Status.WARNING)
                print("COMBINER: Round timed out.")
                # TODO: Generalize policy for what to do in case of timeout. 
                if nr_processed_models >= nr_required_models:
                    break
                else:
                    return None

        self.report_status("ORCHESTRATOR: Training round completed, combined {} models.".format(nr_processed_models),
                           log_level=fedn.Status.INFO)
        self.report_status("DONE, combined {} models".format(nr_processed_models))
        data['nr_successful_updates'] = nr_processed_models
        return model, data

    def __training_round(self,config,clients):

        # We flush the queue at a beginning of a round (no stragglers allowed)
        # TODO: Support other ways to handle stragglers. 
        with self.model_updates.mutex:
            self.model_updates.queue.clear()

        self.report_status("COMBINER: Initiating training round, participating members: {}".format(clients))
        self.server.request_model_update(config['model_id'], clients=clients)

        meta = {}
        meta['nr_expected_updates'] = len(clients)
        meta['nr_required_updates'] = int(config['clients_required'])
        meta['timeout'] = float(config['round_timeout'])
        tic = time.time()
        model = None
        data = None
        try:
            model, data = self.combine_models(nr_expected_models=len(clients), nr_required_models=int(config['clients_required']), timeout=float(config['round_timeout']))
        except Exception as e:
            print("FAILED TO UNPACK FROM COMBINER!",flush=True)
        meta['time_combination'] = time.time()-tic
        meta['aggregation_time'] = data
        return model, meta

    def __validation_round(self,config,clients,model_id):
        self.server.request_model_validation(model_id, clients=clients)

    def stage_model(self,model_id):
        """ Download model from persistent storage. """ 

        # If the model is already in memory at the server we do not need to do anything.
        #TODO ugly ! Need to be refactored
        if self.modelservice.models.exist(model_id):
            return

        # If it is not there, download it from storage and stage it in memory at the server. 
        timeout_retry = 3
        import time
        tries = 0
        while True:
            try:
                model = self.storage.get_model_stream(model_id)
                if model:
                    break
            except Exception as e:
                self.report_status("COMBINER could not fetch model from bucket. retrying in {}".format(timeout_retry),flush=True)
                time.sleep(timeout_retry)
                tries += 1
                if tries > 2:
                    self.report_status("COMBINER exiting. could not fetch seed model.", flush=True)
                    return

        self.modelservice.set_model(model, model_id)

    def __assign_round_clients(self, n):
        """  Obtain a list of clients to talk to in a round. """

        active_trainers = self.server.get_active_trainers()
        # If the number of requested trainers exceeds the number of available, use all available. 
        if n > len(active_trainers):
            n = len(active_trainers)

        # If not, we pick a random subsample of all available clients.
        import random
        clients = random.sample(active_trainers, n)

        return clients

    def __check_nr_round_clients(self, config, timeout=0.0):
        """ Check that the minimal number of required clients to start a round are connected. """

        import time
        ready = False
        t = 0.0
        while not ready:
            active = self.server.nr_active_trainers()

            if active >= int(config['clients_requested']):
                return True
            else:
                self.report_status("waiting for {} clients to get started, currently: {}".format(int(config['clients_requested']) - active,
                                                                                    active), flush=True)
            if t >= timeout:
                if active >= int(config['clients_required']):
                    return True
                else:
                    return False

            time.sleep(1.0)
            t += 1.0

        return ready    

    def exec_validation(self,config,model_id):
        """ Coordinate validation rounds as specified in config. """

        self.report_status("COMBINER orchestrating validation of model {}".format(model_id))
        self.stage_model(model_id)
        #validators = self.__assign_round_clients(int(config['clients_requested']))
        validators = self.__assign_round_clients(self.server.max_clients)
        self.__validation_round(config,validators,model_id)        

    def exec_training(self, config):
        """ Coordinates clients to executee training and validation tasks. """

        round_meta = {}
        round_meta['config'] = config
        round_meta['round_id'] = config['round_id']

        self.stage_model(config['model_id'])

        # Execute the configured number of rounds
        round_meta['local_round'] = {}
        for r in range(1, int(config['rounds']) + 1):
            self.report_status("COMBINER: Starting training round {}".format(r), flush=True)
            #clients = self.__assign_round_clients(int(config['clients_requested']))
            clients = self.__assign_round_clients(self.server.max_clients)
            model, meta = self.__training_round(config, clients)
            round_meta['local_round'][str(r)] = meta
            if model is None:
                self.report_status("\t Failed to update global model in round {0}!".format(r))

        if model is not None:
     
            a = self.helper.serialize_model_to_BytesIO(model)
            # Send aggregated model to server 
            model_id = str(uuid.uuid4())        
            self.modelservice.set_model(a, model_id)
            a.close()
     
            # Update Combiner latest model
            self.server.set_active_model(model_id)

            print("------------------------------------------")
            self.report_status("COMBINER: TRAINING ROUND COMPLETED.", flush=True)
            print("\n")
        return round_meta

    def push_run_config(self, plan):
        self.run_configs_lock.acquire()
        try:
            import uuid
            plan['_job_id'] = str(uuid.uuid4())
            self.run_configs.append(plan)
        finally:
            self.run_configs_lock.release()
        return plan['_job_id']

    def run(self):

        import time
        try:
            while True:
                time.sleep(1)
                #print("combiner run loop...",flush=True)
                #self.server._log_queue_length()
                compute_plan = None
                if len(self.run_configs) > 0:
                    try:
                        self.run_configs_lock.acquire()
                        compute_plan = self.run_configs.pop()
                    finally:
                        self.run_configs_lock.release()
                    self.config = compute_plan
                    self.helper = get_helper(self.config['helper_type'])

                    ready = self.__check_nr_round_clients(compute_plan)
                    if ready:
                        if compute_plan['task'] == 'training':
                            tic = time.time()
                            round_meta = self.exec_training(compute_plan)
                            round_meta['time_exec_training'] = time.time()-tic
                            round_meta['name'] = self.id
                            self.server.tracer.set_round_meta(round_meta)
                        elif compute_plan['task'] == 'validation':
                            self.exec_validation(compute_plan, compute_plan['model_id'])
                        else:
                            self.report_status("COMBINER: Compute plan contains unkown task type.", flush=True)
                    else:
                        self.report_status("COMBINER: Failed to meet client allocation requirements for this compute plan.", flush=True)

               # if self.run_configs_lock.locked():
                #    self.run_configs_lock.release()

        except (KeyboardInterrupt, SystemExit):
            pass