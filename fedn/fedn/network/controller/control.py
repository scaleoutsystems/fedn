import copy
import time
import uuid

from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.controller.controlbase import ControlBase
from fedn.network.state import ReducerState


class UnsupportedStorageBackend(Exception):
    """ Exception class for when storage backend is not supported. Passes """

    def __init__(self, message):
        """ Constructor method.

        :param message: The exception message.
        :type message: str

        """
        self.message = message
        super().__init__(self.message)


class MisconfiguredStorageBackend(Exception):
    """ Exception class for when storage backend is misconfigured. """

    def __init__(self, message):
        """ Constructor method.

        :param message: The exception message.
        :type message: str

        """
        self.message = message
        super().__init__(self.message)

# Exception class for when model is None


class NoModelException(Exception):
    """ Exception class for when model is None """

    def __init__(self, message):
        """ Constructor method.

        :param message: The exception message.
        :type message: str

        """
        self.message = message
        super().__init__(self.message)


class Control(ControlBase):
    """ Controller, implementing the overall global training, validation and inference logic. """

    def __init__(self, statestore):
        """ Constructor method.

        :param statestore: A StateStorage instance.
        :type statestore: class: `fedn.common.storage.statestorage.StateStorage`

        """

        super().__init__(statestore)
        self.name = "DefaultControl"

    def session(self, config):
        """ Execute a new training session. A session consists of one
            or several global rounds. All rounds in the same session
            have the same round_config.

        :param config: The session config.
        :type config: dict

        """

        if self._state == ReducerState.instructing:
            print("Controller already in INSTRUCTING state. A session is in progress.", flush=True)
            return

        self._state = ReducerState.instructing

        # Must be called to set info in the db
        self.new_session(config)

        if not self.get_latest_model():
            print("No model in model chain, please provide a seed model!")

        self._state = ReducerState.monitoring

        last_round = int(self.get_latest_round_id())

        # Execute the rounds in this session
        for round in range(1, int(config['rounds'] + 1)):
            # Increment the round number

            # round_id = self.new_round(session['session_id'])
            if last_round:
                current_round = last_round + round
            else:
                current_round = round

            try:
                _, round_data = self.round(config, current_round)
            except TypeError as e:
                print("Could not unpack data from round: {0}".format(e), flush=True)

            print("CONTROL: Round completed with status {}".format(
                round_data['status']), flush=True)

            self.tracer.set_round_data(round_data)

        # TODO: Report completion of session
        self._state = ReducerState.idle

    def round(self, session_config, round_id):
        """ Execute a single global round.

        :param session_config: The session config.
        :type session_config: dict
        :param round_id: The round id.
        :type round_id: str(int)

        """

        round_data = {'round_id': round_id}

        if len(self.network.get_combiners()) < 1:
            print("REDUCER: No combiners connected!", flush=True)
            round_data['status'] = 'Failed'
            return None, round_data

        # 1. Assemble round config for this global round,
        # and check which combiners are able to participate
        # in the round.
        round_config = copy.deepcopy(session_config)
        round_config['rounds'] = 1
        round_config['round_id'] = round_id
        round_config['task'] = 'training'
        round_config['model_id'] = self.get_latest_model()
        round_config['helper_type'] = self.statestore.get_helper()

        combiners = self.get_participating_combiners(round_config)
        round_start = self.evaluate_round_start_policy(combiners)

        if round_start:
            print("CONTROL: round start policy met, participating combiners {}".format(
                combiners), flush=True)
        else:
            print("CONTROL: Round start policy not met, skipping round!", flush=True)
            round_data['status'] = 'Failed'
            return None

        round_data['round_config'] = round_config

        # 2. Ask participating combiners to coordinate model updates
        _ = self.request_model_updates(combiners)

        # Wait until participating combiners have produced an updated global model.
        wait = 0.0
        # dict to store combiners that have successfully produced an updated model
        updated = {}
        # wait until all combiners have produced an updated model or until round timeout
        print("CONTROL: Fetching round config (ID: {round_id}) from statestore:".format(
            round_id=round_id), flush=True)
        while len(updated) < len(combiners):
            round = self.statestore.get_round(round_id)
            if round:
                print("CONTROL: Round found!", flush=True)
                # For each combiner in the round, check if it has produced an updated model (status == 'Success')
                for combiner in round['combiners']:
                    print(combiner, flush=True)
                    if combiner['status'] == 'Success':
                        if combiner['name'] not in updated.keys():
                            # Add combiner to updated dict
                            updated[combiner['name']] = combiner['model_id']
                    # Print combiner status
                    print("CONTROL: Combiner {name} status: {status}".format(
                        name=combiner['name'], status=combiner['status']), flush=True)
            else:
                # Print every 10 seconds based on value of wait
                if wait % 10 == 0:
                    print("CONTROL: Round not found! Waiting...", flush=True)
            if wait >= session_config['round_timeout']:
                print("CONTROL: Round timeout! Exiting round...", flush=True)
                break
            # Update wait time used for timeout
            time.sleep(1.0)
            wait += 1.0

        round_valid = self.evaluate_round_validity_policy(updated)
        if not round_valid:
            print("REDUCER CONTROL: Round invalid!", flush=True)
            round_data['status'] = 'Failed'
            return None, round_data

        print("CONTROL: Reducing models from combiners...", flush=True)
        # 3. Reduce combiner models into a global model
        try:
            model, data = self.reduce(updated)
            round_data['reduce'] = data
            print("CONTROL: Done reducing models from combiners!", flush=True)
        except Exception as e:
            print("CONTROL: Failed to reduce models from combiners: {}".format(
                e), flush=True)
            round_data['status'] = 'Failed'
            return None, round_data

        # 6. Commit the global model to model trail
        if model is not None:
            print("CONTROL: Committing global model to model trail...", flush=True)
            tic = time.time()
            model_id = uuid.uuid4()
            self.commit(model_id, model)
            round_data['time_commit'] = time.time() - tic
            print("CONTROL: Done committing global model to model trail!", flush=True)
        else:
            print("REDUCER: failed to update model in round with config {}".format(
                session_config), flush=True)
            round_data['status'] = 'Failed'
            return None, round_data

        round_data['status'] = 'Success'

        # 4. Trigger participating combiner nodes to execute a validation round for the current model
        validate = session_config['validate']
        if validate:
            combiner_config = copy.deepcopy(session_config)
            combiner_config['round_id'] = round_id
            combiner_config['model_id'] = self.get_latest_model()
            combiner_config['task'] = 'validation'
            combiner_config['helper_type'] = self.statestore.get_helper()

            validating_combiners = self._select_participating_combiners(
                combiner_config)

            for combiner, combiner_config in validating_combiners:
                try:
                    print("CONTROL: Submitting validation round to combiner {}".format(
                        combiner), flush=True)
                    combiner.submit(combiner_config)
                except CombinerUnavailableError:
                    self._handle_unavailable_combiner(combiner)
                    pass

        return model_id, round_data

    def reduce(self, combiners):
        """ Combine updated models from Combiner nodes into one global model.

        :param combiners: dict of combiner names (key) and model IDs (value) to reduce
        :type combiners: dict
        """

        meta = {}
        meta['time_fetch_model'] = 0.0
        meta['time_load_model'] = 0.0
        meta['time_aggregate_model'] = 0.0

        i = 1
        model = None
        # Check if there are any combiners to reduce
        if len(combiners) == 0:
            print("REDUCER: No combiners to reduce!", flush=True)
            return model, meta

        for name, model_id in combiners.items():

            # TODO: Handle inactive RPC error in get_model and raise specific error
            print("REDUCER: Fetching model ({model_id}) from combiner {name}".format(
                model_id=model_id, name=name), flush=True)
            try:
                tic = time.time()
                combiner = self.get_combiner(name)
                data = combiner.get_model(model_id)
                meta['time_fetch_model'] += (time.time() - tic)
            except Exception as e:
                print("REDUCER: Failed to fetch model from combiner {}: {}".format(
                    name, e), flush=True)
                data = None

            if data is not None:
                try:
                    tic = time.time()
                    helper = self.get_helper()
                    data.seek(0)
                    model_next = helper.load(data)
                    meta['time_load_model'] += (time.time() - tic)
                    tic = time.time()
                    model = helper.increment_average(model, model_next, i, i)
                    meta['time_aggregate_model'] += (time.time() - tic)
                except Exception:
                    tic = time.time()
                    data.seek(0)
                    model = helper.load(data)
                    meta['time_aggregate_model'] += (time.time() - tic)
                i = i + 1

        return model, meta

    def infer_instruct(self, config):
        """ Main entrypoint for executing the inference compute plan.

        :param config: configuration for the inference round
        """

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
        """ Execute an inference round.

        :param config: configuration for the inference round
        """

        # Init meta
        round_data = {}

        # Check for at least one combiner in statestore
        if len(self.network.get_combiners()) < 1:
            print("REDUCER: No combiners connected!")
            return round_data

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
                combiner.submit(combiner_config)
            except CombinerUnavailableError:
                # It is OK if inference fails for a combiner
                self._handle_unavailable_combiner(combiner)
                pass

        return round_data
