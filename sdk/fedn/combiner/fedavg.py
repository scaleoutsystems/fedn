import json
import os
import queue
import tempfile
import time

import fedn.proto.alliance_pb2 as alliance
import tensorflow as tf
from fedn.combiner.server import CombinerClient
from fedn.utils.helpers import KerasSequentialHelper
#from fedn.utils.mongo import connect_to_mongodb

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class FEDAVGCombiner(CombinerClient):
    """ A Local SGD / Federated Averaging (FedAvg) combiner. """

    def __init__(self, address, port, id, role, storage):

        super().__init__(address, port, id, role)

        self.storage = storage
#        self.id = id

        self.config = {}
        # TODO: Use MongoDB
        self.validations = {}

        # Default helper, can be overriden via config at runtime
        self.helper = KerasSequentialHelper()

        # Queue for model updates to be processed.
        self.model_updates = queue.Queue()

    def get_model_id(self):
        return self.model_id

    def receive_model_candidate(self, model_id):
        """ Callback when a new model version is reported by a client. 
            We simply put the model_id on a queue to be processed later. """
        try:
            self.report_status("COMBINER: callback received model {}".format(model_id),
                               log_level=alliance.Status.INFO)
            # TODO - here would be a place to do some additional validation of the model contribution. 
            self.model_updates.put(model_id)
        except Exception as e:
            self.report_status("COMBINER: Failed to receive candidate model! {}".format(e),
                               log_level=alliance.Status.WARNING)
            print("Failed to receive candidate model!")
            pass

    def receive_validation(self, validation):
        """ Callback for a validation request """

        # TODO: Track this in a DB
        model_id = validation.model_id
        data = json.loads(validation.data)
        try:
            self.validations[model_id].append(data)
        except KeyError:
            self.validations[model_id] = [data]

        self.report_status("COMBINER: callback processed validation {}".format(validation.model_id),
                           log_level=alliance.Status.INFO)

    def combine_models(self, nr_expected_models=None, timeout=120):
        """ Compute an iterative/running average of models arriving to the combiner. """

        round_time = 0.0
        print("COMBINER: combining model updates...")

        # First model in the update round
        try:
            model_id = self.model_updates.get(timeout=timeout)
            print("combining ", model_id)
            # Fetch the model data blob from storage
            model_str = self.get_model(model_id)
            model = self.helper.load_model(model_str.getbuffer())
            nr_processed_models = 1
            self.model_updates.task_done()
        except queue.Empty as e:
            self.report_status("COMBINER: training round timed out.", log_level=alliance.Status.WARNING)
            return None

        while nr_processed_models < nr_expected_models:
            try:
                model_id = self.model_updates.get(block=False)
                self.report_status("Received model update with id {}".format(model_id))

                model_next = self.helper.load_model(self.get_model(model_id).getbuffer())
                self.helper.increment_average(model, model_next, nr_processed_models)

                nr_processed_models += 1
                self.model_updates.task_done()
            except Exception as e:
                self.report_status("COMBINER failcode: {}".format(e))
                time.sleep(1.0)
                round_time += 1.0

            if round_time >= timeout:
                self.report_status("COMBINER: training round timed out.", log_level=alliance.Status.WARNING)
                print("COMBINER: Round timed out.")
                return None

        self.report_status("ORCHESTRATOR: Training round completed, combined {} models.".format(nr_processed_models),
                           log_level=alliance.Status.INFO)
        print("DONE, combined {} models".format(nr_processed_models))
        return model


    def __training_round(self,config,clients):

        # We flush the queue at a beginning of a round (no stragglers allowed)
        # TODO: Support other ways to handle stragglers. 
        with self.model_updates.mutex:
            self.model_updates.queue.clear()

        self.report_status("COMBINER: Initiating training round, participating members: {}".format(clients))
        self.request_model_update(config['model_id'], clients=clients)

        # Apply combiner
        model = self.combine_models(nr_expected_models=len(clients), timeout=config['round_timeout'])
        return model

    def __validation_round(self,config,clients):
        self.request_model_validation(config['model_id'], from_clients=clients)


    def run_training_rounds(self,config):
        """ Coordinate training rounds as specified in config. """

        # Add here to override default helpers
        if config['ml_framework'] == 'Keras':
            self.helper = KerasSequentialHelper()

        self._stage_active_model(config['model_id'])
        self.model_id = config['model_id']

        print("COMBINER starting from model {}".format(self.model_id))
        
        # Check that the minimal number of required clients to start a round are connected 
        ready = self._check_nr_round_clients(config['clients_required'])

        result = {}

        # Execute the configured number of rounds
        for r in range(1, config['rounds'] + 1):

            print("STARTING ROUND {}".format(r), flush=True)
            print("\t FEDAVG: Starting training round {}".format(r), flush=True)

            # Ask clients to update the model
            trainers = self._assign_round_clients(config['clients_requested'])
            model = self.__training_round(config,trainers)

            if model:
                print("\t FEDAVG: Round completed.", flush=True)

                # TODO: Use  helper to serialize model - this is a Keras specific solution
                fod, outfile_name = tempfile.mkstemp(suffix='.h5')
                model.save(outfile_name)

                # Upload new model to storage repository (persistent)
                # and save to local storage for sharing with clients.

                # TODO: Make checkpointing in persistent storage a configurable option
                model_id = self.storage.set_model(outfile_name, is_file=True)

                from io import BytesIO
                a = BytesIO()
                with open(outfile_name,'rb') as f:
                    a.write(f.read())

                # Stream aggregated model to server 
                # TODO: Not strictly necessary to stream model here, can be slight waste of resources.
                self.set_model(a, model_id) 
                os.unlink(outfile_name)

                self.model_id = model_id

                print("...done. New aggregated model: {}".format(self.model_id))
                print("------------------------------------------")
                print("FEDAVG: ROUND COMPLETED.", flush=True)
                print("\n")
            else:
                print("\t Failed to update global model in round {0}!".format(r))

        result['model_id'] = self.model_id
        return result

    def run_validation(self,config):
        """ Coordinate validation rounds as specified in config. """

        print("COMBINER orchestrating validation of model {}".format(config['model_id']))
        ready = self._check_nr_round_clients(config['clients_required'])
                
        validators = self._assign_round_clients(config['clients_requested'])
        self.__validation_round(config,validators)

    def run(self,config):
        result = self.run_training_rounds(config)
        config['model_id'] = result['model_id']
        self.run_validation(config)
