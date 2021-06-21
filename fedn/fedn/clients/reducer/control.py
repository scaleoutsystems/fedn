import copy
import os
import tempfile
import time

from fedn.clients.reducer.interfaces import CombinerUnavailableError
from fedn.clients.reducer.network import Network
from .state import ReducerState

import fedn.utils.helpers


class UnsupportedStorageBackend(Exception):
    pass

class MisconfiguredStorageBackend(Exception):
    pass

class ReducerControl:

    def __init__(self, statestore):
        self.__state = ReducerState.setup
        self.statestore = statestore
        if self.statestore.is_inited():
            self.network = Network(self, statestore)

        try:
            config = self.statestore.get_storage_backend()
        except:
            print("REDUCER CONTROL: Failed to retrive storage configuration, exiting.",flush=True)
            raise MisconfiguredStorageBackend()
        if not config:
            print("REDUCER CONTROL: No storage configuration available, exiting.",flush=True)
            raise MisconfiguredStorageBackend()

        if config['storage_type'] == 'S3':
            from fedn.common.storage.s3.s3repo import S3ModelRepository
            self.model_repository = S3ModelRepository(config['storage_config'])
        else:
            print("REDUCER CONTROL: Unsupported storage backend, exiting.",flush=True)
            raise UnsupportedStorageBackend()

        self.client_allocation_policy = self.client_allocation_policy_least_packed

        if self.statestore.is_inited():
            self.__state = ReducerState.idle

    def get_helper(self):
        helper_type = self.statestore.get_framework()
        helper = fedn.utils.helpers.get_helper(helper_type)
        if not helper:
            print("CONTROL: Unsupported helper type {}, please configure compute_context.helper !".format(helper_type),flush=True)
            return None
        return helper

    def delete_bucket_objects(self):
        return self.model_repository.delete_objects()

    def get_state(self):
        return self.__state

    def idle(self):
        if self.__state == ReducerState.idle:
            return True
        else:
            return False

    def get_first_model(self):
        return self.statestore.get_first()

    def get_latest_model(self):
        return self.statestore.get_latest()

    def get_model_info(self):
        return self.statestore.get_model_info()

    def get_events(self):
        return self.statestore.get_events()

    def drop_models(self):
        self.statestore.drop_models()

    def get_compute_context(self):
        definition = self.statestore.get_compute_context()
        if definition:
            try:
                context = definition['filename']
                return context
            except (IndexError, KeyError):
                print("No context filename set for compute context definition", flush=True)
                return None
        else:
            return None

    def set_compute_context(self, filename, path):
        """ Persist the configuration for the compute package. """
        self.model_repository.set_compute_context(filename, path)
        self.statestore.set_compute_context(filename)

    def get_compute_package(self, compute_package=''):
        if compute_package == '':
            compute_package = self.get_compute_context()
        return self.model_repository.get_compute_package(compute_package)


    def commit(self, model_id, model=None):
        """ Commit a model to the global model trail. The model commited becomes the lastest consensus model. """

        helper = self.get_helper()
        if model is not None:
            print("Saving model to disk...",flush=True)
            outfile_name = helper.save_model(model)
            print("DONE",flush=True)
            print("Uploading model to Minio...",flush=True)
            model_id = self.model_repository.set_model(outfile_name, is_file=True)
            print("DONE",flush=True)
            os.unlink(outfile_name)

        self.statestore.set_latest(model_id)

    def _out_of_sync(self,combiners=None):

        if not combiners:
            combiners = self.network.get_combiners()

        osync = []
        for combiner in combiners:
            try:
                model_id = combiner.get_model_id()
                print("COMBINER UPDATE, model ID", model_id, flush=True)
            except CombinerUnavailableError:
                self._handle_unavailable_combiner(combiner)
                model_id = None
                raise
            if model_id and (model_id != self.get_latest_model()):
                osync.append(combiner)
        return osync

    def check_round_participation_policy(self,compute_plan,combiner_state):
        """ Evaluate reducer level policy for combiner round-paarticipation.
            This is a decision on ReducerControl level, additional checks
            applies on combiner level. Not all reducer control flows might
            need or want to use a participation policy.  """

        if int(compute_plan['clients_required']) <= int(combiner_state['nr_active_clients']):
            return True
        else:
            return False

    def check_round_start_policy(self,combiners):
        """ Check if the overall network state meets the policy to start a round. """
        if len(combiners) > 0:
            return True
        else:
            return False

    def check_round_validity_policy(self,combiners):
        """
            At the end of the round, before committing a model to the model ledger,
            we check if a round validity policy has been met. This can involve
            e.g. asserting that a certain number of combiners have reported in an
            updated model, or that criteria on model performance have been met.
        """
        if combiners == []:
            return False
        else:
            return True

    def _handle_unavailable_combiner(self,combiner):
        """ This callback is triggered if a combiner is found to be unresponsive. """
        # TODO: Implement strategy to handle the case.
        print("REDUCER CONTROL: Combiner {} unavailable.".format(combiner.name),flush=True)

    def round(self, config, round_number):
        """ Execute one global round. """

        round_meta = {'round_id': round_number}

        # TODO: Set / update reducer states and such
        # TODO: Do a General Health check on Combiners in the beginning of the round.
        if len(self.network.get_combiners()) < 1:
            print("REDUCER: No combiners connected!")
            return None

        # 1. Formulate compute plans for this round and determine which combiners should participate in the round.
        compute_plan = copy.deepcopy(config)
        compute_plan['rounds'] = 1
        compute_plan['round_id'] = round_number
        compute_plan['task'] = 'training'
        compute_plan['model_id'] = self.get_latest_model()
        compute_plan['helper_type'] = self.statestore.get_framework()

        round_meta['compute_plan'] = compute_plan

        combiners = []
        for combiner in self.network.get_combiners():

            try:
                combiner_state = combiner.report()
            except CombinerUnavailableError:
                self._handle_unavailable_combiner(combiner)
                combiner_state = None

            if combiner_state:
                is_participating = self.check_round_participation_policy(compute_plan,combiner_state)
                if is_participating:
                    combiners.append((combiner,compute_plan))


        round_start = self.check_round_start_policy(combiners)
        print("CONTROL: round start policy met, participating combiners {}".format(round_start),flush=True)
        if not round_start:
            print("CONTROL: Round start policy not met, skipping round!",flush=True)
            return None

        # 2. Sync up and ask participating combiners to coordinate model updates
        # TODO refactor
        from datetime import datetime
        from fedn.common.tracer.mongotracer import MongoTracer
        statestore_config = self.statestore.get_config()

        self.tracer = MongoTracer(statestore_config['mongo_config'], statestore_config['network_id'])

        start_time = datetime.now()

        for combiner,compute_plan in combiners:
            try:
                self.sync_combiners([combiner],self.get_latest_model())
                response = combiner.start(compute_plan)
            except CombinerUnavailableError:
                # This is OK, handled by round accept policy
                self._handle_unavailable_combiner(combiner)
                pass
            except:
                # Unknown error
                raise

        # Wait until participating combiners have a model that is out of sync with the current global model.
        # TODO: We do not need to wait until all combiners complete before we start reducing.
        cl = []
        for combiner,plan in combiners:
            cl.append(combiner)

        wait = 0.0
        while len(self._out_of_sync(cl)) < len(combiners):
            time.sleep(1.0)
            wait += 1.0
            if wait >= config['round_timeout']:
                break

        # TODO refactor
        end_time = datetime.now()
        round_time = end_time - start_time
        self.tracer.set_combiner_time(round_number, round_time.seconds)

        round_meta['time_combiner_update'] = round_time.seconds

        # OBS! Here we are checking against all combiners, not just those that computed in this round.
        # This means we let straggling combiners participate in the update
        updated = self._out_of_sync()
        print("COMBINERS UPDATED MODELS: {}".format(updated),flush=True)

        print("Checking round validity policy...",flush=True)
        round_valid = self.check_round_validity_policy(updated)
        if round_valid == False:
            # TODO: Should we reset combiner state here?
            print("REDUCER CONTROL: Round invalid!",flush=True)
            return None, round_meta
        print("OK")

        print("Starting reducing models...",flush=True)
        # 3. Reduce combiner models into a global model
        try:
            model,data = self.reduce(updated)
            round_meta['reduce'] = data
        except Exception as e:
            print("CONTROL: Failed to reduce models from combiners: {}".format(updated),flush=True)
            print(e,flush=True)
            return None, round_meta
        print("DONE",flush=True)


        # 6. Commit the global model to the ledger
        print("Committing global model...",flush=True)
        if model is not None:
            # Commit to model ledger
            tic = time.time()
            import uuid
            model_id = uuid.uuid4()
            self.commit(model_id,model)
            round_meta['time_commit']=time.time()-tic
        else:
            print("REDUCER: failed to update model in round with config {}".format(config),flush=True)
            return None, round_meta
        print("DONE",flush=True)

        # 4. Trigger participating combiner nodes to execute a validation round for the current model
        validate = config['validate']
        if validate:
            combiner_config = copy.deepcopy(config)
            combiner_config['model_id'] = self.get_latest_model()
            combiner_config['task'] = 'validation'
            combiner_config['helper_type'] = self.statestore.get_framework()

            for combiner in updated:
                try:
                    combiner.start(combiner_config)
                except CombinerUnavailableError:
                    # OK if validation fails for a combiner
                    self._handle_unavailable_combiner(combiner)
                    pass

        # 5. Check commit policy based on validation result (optionally)
        # TODO: Implement.
                


        return model_id, round_meta

    def sync_combiners(self, combiners, model_id):
        """ Spread the current consensus model to all active combiner nodes. """
        if not model_id:
            print("GOT NO MODEL TO SET! Have you seeded the FedML model?", flush=True)
            return

        for combiner in combiners:
            response = combiner.set_model_id(model_id)

    def instruct(self, config):
        """ Main entrypoint, executes the compute plan. """

        if self.__state == ReducerState.instructing:
            print("Already set in INSTRUCTING state", flush=True)
            return

        self.__state = ReducerState.instructing

        if not self.get_latest_model():
            print("No model in model chain, please seed the alliance!")

        self.__state = ReducerState.monitoring

        # TODO: Validate and set the round config object
        #self.set_config(config)

        # TODO: Refactor
        from fedn.common.tracer.mongotracer import MongoTracer
        statestore_config = self.statestore.get_config()
        self.tracer = MongoTracer(statestore_config['mongo_config'], statestore_config['network_id'])
        last_round = self.tracer.get_latest_round()

        for round in range(1, int(config['rounds'] + 1)):
            tic = time.time()
            if last_round:
                current_round = last_round + round
            else:
                current_round = round

            from datetime import datetime
            start_time = datetime.now()
            # start round monitor
            self.tracer.start_monitor(round)
            # todo add try except bloc for round meta
            model_id = None
            round_meta = {'round_id':current_round}
            try:
                model_id, round_meta = self.round(config, current_round)
            except TypeError:
                print("Could not unpack data from round...", flush=True)

            end_time = datetime.now()
            
            if model_id:
                print("REDUCER: Global round completed, new model: {}".format(model_id), flush=True)
                round_time = end_time - start_time
                self.tracer.set_latest_time(current_round, round_time.seconds)
                round_meta['status'] = 'Success'
            else:
                print("REDUCER: Global round failed!")
                round_meta['status'] = 'Failed'

            # stop round monitor
            self.tracer.stop_monitor()
            round_meta['time_round'] = time.time()-tic
            self.tracer.set_round_meta_reducer(round_meta)


        self.__state = ReducerState.idle

    def reduce(self, combiners):
        """ Combine current models at Combiner nodes into one global model. """

        meta = {}
        meta['time_fetch_model'] = 0.0
        meta['time_load_model'] = 0.0
        meta['time_aggregate_model'] = 0.0

        i = 1
        model = None
        for combiner in combiners:

            # TODO: Handle inactive RPC error in get_model and raise specific error
            try:
                tic = time.time()
                data = combiner.get_model()
                meta['time_fetch_model'] += (time.time()-tic)
            except:
                pass

            helper = self.get_helper()

            if data is not None:
                try:
                    tic = time.time()
                    model_str=combiner.get_model().getbuffer()
                    model_next = helper.load_model_from_BytesIO(model_str)
                    meta['time_load_model'] += (time.time()-tic)
                    tic = time.time()
                    model = helper.increment_average(model, model_next, i)
                    meta['time_aggregate_model'] += (time.time()-tic)
                except:
                    tic = time.time()
                    model = helper.load_model_from_BytesIO(data.getbuffer())
                    meta['time_aggregate_model'] += (time.time()-tic)
                i = i+1

        return model, meta

    def monitor(self, config=None):
        #status = self.network.check_health()
        pass

    def client_allocation_policy_first_available(self):
        """
            Allocate client to the first available combiner in the combiner list.
            Packs one combiner full before filling up next combiner.
        """
        for combiner in self.network.get_combiners():
            if combiner.allowing_clients():
                return combiner
        return None

    def client_allocation_policy_least_packed(self):
        """
            Allocate client to the available combiner with the smallest number of clients.
            Spreads clients evenly over all active combiners.

            TODO: Not thread safe - not garanteed to result in a perfectly even partition.

        """
        min_clients = None
        selected_combiner = None

        for combiner in self.network.get_combiners():
            try:
                if combiner.allowing_clients():
                    combiner_state = combiner.report()
                    nac = combiner_state['nr_active_clients']
                    if not min_clients:
                        min_clients = nac
                        selected_combiner = combiner
                    elif nac<min_clients:
                        min_clients = nac
                        selected_combiner = combiner
            except CombinerUnavailableError as err:
                print("Combiner was not responding, continuing to next")

        return selected_combiner

    def find(self, name):
        for combiner in self.network.get_combiners():
            if name == combiner.name:
                return combiner
        return None

    def find_available_combiner(self):
        combiner = self.client_allocation_policy()
        return combiner

    def state(self):
        return self.__state
