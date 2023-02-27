import copy
import time
import uuid

from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.controller.controlbase import ControlBase
from fedn.network.state import ReducerState


class UnsupportedStorageBackend(Exception):
    pass


class MisconfiguredStorageBackend(Exception):
    pass


class Control(ControlBase):
    """ Controller, implementing the overall global training strategy.

    """

    def __init__(self, statestore):

        super().__init__(statestore)
        self.name = "DefaultControl"

    def session(self, config):
        """ Execute a new training session. A session consists of one
            or several global rounds. All rounds in the same session
            have the same round_config
        """

        if self._state == ReducerState.instructing:
            print("Controller already in INSTRUCTING state. A session is in progress.", flush=True)
            return

        self._state = ReducerState.instructing

        if not self.get_latest_model():
            print("No model in model chain, please provide a seed model!")

        if "session_id" not in config.keys():
            session_id = uuid.uuid4()
            config['session_id'] = str(session_id)

        # self.statestore.new_session(session_id)
        self._state = ReducerState.monitoring

        last_round = int(self.get_latest_round_id())

        # Execute the rounds in this session
        for round in range(1, int(config['rounds'] + 1)):
            # Increment the round number
            if last_round:
                current_round = last_round + round
            else:
                current_round = round

            model_id = None
            round_meta = {'round_id': current_round}
            round_meta['session_id'] = session_id

            try:
                model_id, round_meta = self.round(config, current_round)
            except TypeError:
                raise
                print("Could not unpack data from round...", flush=True)

            if model_id:
                print("CONTROL: Round completed, new global model: {}".format(
                    model_id), flush=True)
                round_meta['status'] = 'Success'
            else:
                print("CONTROL: Round failed!")
                round_meta['status'] = 'Failed'

            self.tracer.set_round_meta_reducer(round_meta)

        # TODO: Report completion of session
        self._state = ReducerState.idle

    def round(self, session_config, round_number):
        """Execute one round. """

        round_meta = {'round_id': round_number}

        if len(self.network.get_combiners()) < 1:
            print("REDUCER: No combiners connected!", flush=True)
            return None, round_meta

        # 1. Assemble round config for this global round,
        # and check which combiners are able to participate
        # in the round.
        round_config = copy.deepcopy(session_config)
        round_config['rounds'] = 1
        round_config['round_id'] = round_number
        round_config['task'] = 'training'
        round_config['model_id'] = self.get_latest_model()
        round_config['helper_type'] = self.statestore.get_framework()

        combiners = self.get_participating_combiners(round_config)
        round_start = self.evaluate_round_start_policy(combiners)

        if round_start:
            print("CONTROL: round start policy met, participating combiners {}".format(
                combiners), flush=True)
        else:
            print("CONTROL: Round start policy not met, skipping round!", flush=True)
            return None

        round_meta['round_config'] = round_config

        # 2. Ask participating combiners to coordinate model updates
        cl = self.request_model_updates(combiners)

        # Wait until participating combiners have produced an updated global model.
        # TODO: Refactor
        wait = 0.0
        updated = {}
        while len(updated) < len(combiners):
            round = self.statestore.get_round(round_number)
            if round:
                for combiner in round['combiners']:
                    print(combiner, flush=True)
                    if combiner['status'] == 'Success':
                        if combiner['name'] not in updated.keys():
                            updated[combiner['name']] = combiner['model_id']
                    print(combiner['status'])

            if wait >= session_config['round_timeout']:
                break
            time.sleep(1.0)
            wait += 1.0
        print(updated, flush=True)

        print("Checking round validity policy...", flush=True)
        round_valid = self.evaluate_round_validity_policy(updated)
        if not round_valid:
            print("REDUCER CONTROL: Round invalid!", flush=True)
            return None, round_meta
        print("Round valid.", flush=True)

        print("Starting reducing models...", flush=True)
        # 3. Reduce combiner models into a global model
        try:
            model, data = self.reduce(updated)
            round_meta['reduce'] = data
        except Exception as e:
            print("CONTROL: Failed to reduce models from combiners: {}".format(
                updated), flush=True)
            print(e, flush=True)
            return None, round_meta
        print("DONE", flush=True)

        # 6. Commit the global model to model trail
        if model is not None:
            print("Committing global model...", flush=True)
            tic = time.time()
            model_id = uuid.uuid4()
            self.commit(model_id, model)
            round_meta['time_commit'] = time.time() - tic
        else:
            print("REDUCER: failed to update model in round with config {}".format(
                session_config), flush=True)
            return None, round_meta
        print("DONE", flush=True)

        # 4. Trigger participating combiner nodes to execute a validation round for the current model
        validate = session_config['validate']
        if validate:
            combiner_config = copy.deepcopy(session_config)
            combiner_config['round_id'] = round_number
            combiner_config['model_id'] = self.get_latest_model()
            combiner_config['task'] = 'validation'
            combiner_config['helper_type'] = self.statestore.get_framework()

            validating_combiners = self._select_participating_combiners(
                combiner_config)

            for combiner, combiner_config in validating_combiners:
                try:
                    #self.set_combiner_model([combiner], self.get_latest_model())
                    combiner.start(combiner_config)
                except CombinerUnavailableError:
                    self._handle_unavailable_combiner(combiner)
                    pass

        return model_id, round_meta

    def reduce(self, combiners):
        """ Combine current models at Combiner nodes into one global model. """

        meta = {}
        meta['time_fetch_model'] = 0.0
        meta['time_load_model'] = 0.0
        meta['time_aggregate_model'] = 0.0

        i = 1
        model = None
        for name, model_id in combiners.items():

            # TODO: Handle inactive RPC error in get_model and raise specific error
            try:
                tic = time.time()
                combiner = self.get_combiner(name)
                data = combiner.get_model(model_id)
                meta['time_fetch_model'] += (time.time() - tic)
            except Exception:
                data = None

            if data is not None:
                try:
                    tic = time.time()
                    model_str = data.getbuffer()
                    helper = self.get_helper()
                    model_next = helper.load_model_from_BytesIO(model_str)
                    meta['time_load_model'] += (time.time() - tic)
                    tic = time.time()
                    model = helper.increment_average(model, model_next, i)
                    meta['time_aggregate_model'] += (time.time() - tic)
                except Exception:
                    tic = time.time()
                    model = helper.load_model_from_BytesIO(data.getbuffer())
                    meta['time_aggregate_model'] += (time.time() - tic)
                i = i + 1

        return model, meta

    def infer_instruct(self, config):
        """ Main entrypoint for executing the inference compute plan. """

        # Check/set instucting state
        if self.__state == ReducerState.instructing:
            print("Already set in INSTRUCTING state", flush=True)
            return
        self.__state = ReducerState.instructing

        # Check for a model chain
        if not self.get_latest_model():
            print("No model in model chain, please seed the alliance!")

        # Set reducer in monitoring state
        self.__state = ReducerState.monitoring

        # Start inference round
        try:
            self.inference_round(config)
        except TypeError:
            print("Could not unpack data from round...", flush=True)

        # Set reducer in idle state
        self.__state = ReducerState.idle

    def inference_round(self, config):
        """ Execute inference round. """

        # Init meta
        round_meta = {}

        # Check for at least one combiner in statestore
        if len(self.network.get_combiners()) < 1:
            print("REDUCER: No combiners connected!")
            return round_meta

        # Setup combiner configuration
        combiner_config = copy.deepcopy(config)
        combiner_config['model_id'] = self.get_latest_model()
        combiner_config['task'] = 'inference'
        combiner_config['helper_type'] = self.statestore.get_framework()

        # Select combiners
        validating_combiners = self._select_round_combiners(
            combiner_config)

        # Test round start policy
        round_start = self.check_round_start_policy(validating_combiners)
        if round_start:
            print("CONTROL: round start policy met, participating combiners {}".format(
                validating_combiners), flush=True)
        else:
            print("CONTROL: Round start policy not met, skipping round!", flush=True)
            return None

        # Synch combiners with latest model and trigger inference
        for combiner, combiner_config in validating_combiners:
            try:
                # TODO: FIX HERE.
                self.sync_combiners([combiner], self.get_latest_model())
                combiner.start(combiner_config)
            except CombinerUnavailableError:
                # It is OK if inference fails for a combiner
                self._handle_unavailable_combiner(combiner)
                pass

        return round_meta
