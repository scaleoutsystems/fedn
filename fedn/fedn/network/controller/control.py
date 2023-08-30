import copy
import datetime
import time
import uuid

from tenacity import (retry, retry_if_exception_type, stop_after_delay,
                      wait_random)

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
    """ Exception class for when storage backend is misconfigured.

    :param message: The exception message.
    :type message: str
    """

    def __init__(self, message):
        """ Constructor method."""
        self.message = message
        super().__init__(self.message)


class NoModelException(Exception):
    """ Exception class for when model is None

    :param message: The exception message.
    :type message: str
    """

    def __init__(self, message):
        """ Constructor method."""
        self.message = message
        super().__init__(self.message)


class CombinersNotDoneException(Exception):
    """ Exception class for when model is None """

    def __init__(self, message):
        """ Constructor method.

        :param message: The exception message.
        :type message: str

        """
        self.message = message
        super().__init__(self.message)


class Control(ControlBase):
    """ Controller, implementing the overall global training, validation and inference logic.

    :param statestore: A StateStorage instance.
    :type statestore: class: `fedn.network.statestorebase.StateStorageBase`
    """

    def __init__(self, statestore):
        """ Constructor method."""

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

        if not self.statestore.get_latest_model():
            print("No model in model chain, please provide a seed model!")
            return

        self._state = ReducerState.instructing

<<<<<<< HEAD
        # Must be called once to set info in the db
=======
        # Must be called to set info in the db
        config['committed_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
>>>>>>> develop
        self.new_session(config)

        if not self.statestore.get_latest_model():
            print("No model in model chain, please provide a seed model!", flush=True)
        self._state = ReducerState.monitoring

        last_round = int(self.get_latest_round_id())

        # Clear potential stragglers/old model updates at combiners
        for combiner in self.network.get_combiners():
            combiner.flush_model_update_queue()

        # Execute the rounds in this session
        for round in range(1, int(config['rounds'] + 1)):
            # Increment the round number
<<<<<<< HEAD
=======

>>>>>>> develop
            if last_round:
                current_round = last_round + round
            else:
                current_round = round

            try:
                _, round_data = self.round(config, str(current_round))
            except TypeError as e:
                print("Could not unpack data from round: {0}".format(e), flush=True)

            print("CONTROL: Round completed with status {}".format(
                round_data['status']), flush=True)

        # TODO: Report completion of session
        self._state = ReducerState.idle

    def round(self, session_config, round_id):
        """ Execute one global round.

        :param session_config: The session config.
        :type session_config: dict
        :param round_id: The round id.
<<<<<<< HEAD
        :type round_id: str

=======
        :type round_id: str(int)
>>>>>>> develop
        """

        self.new_round({'round_id': round_id, 'status': "Pending"})

        if len(self.network.get_combiners()) < 1:
            print("CONTROLLER: Round cannot start, no combiners connected!", flush=True)
            self.set_round_status(round_id, 'Failed')
            return None, self.statestore.get_round(round_id)

        # Assemble round config for this global round
        round_config = copy.deepcopy(session_config)
        round_config['rounds'] = 1
        round_config['round_id'] = round_id
        round_config['task'] = 'training'
        round_config['model_id'] = self.statestore.get_latest_model()
        round_config['helper_type'] = self.statestore.get_helper()

        self.set_round_config(round_id, round_config)

        # Get combiners that are able to participate in round, given round_config
        participating_combiners = self.get_participating_combiners(round_config)

        # Check if the policy to start the round is met
        round_start = self.evaluate_round_start_policy(participating_combiners)

        if round_start:
            print("CONTROL: round start policy met, {} participating combiners.".format(
                len(participating_combiners)), flush=True)
        else:
            print("CONTROL: Round start policy not met, skipping round!", flush=True)
            self.set_round_status(round_id, 'Failed')
            return None, self.statestore.get_round(round_id)

        # Ask participating combiners to coordinate model updates
        _ = self.request_model_updates(participating_combiners)
        # TODO: Check response

        # Wait until participating combiners have produced an updated global model,
        # or round times out.
        def do_if_round_times_out(result):
            print("CONTROL: Round timed out!", flush=True)

        @retry(wait=wait_random(min=1.0, max=2.0),
               stop=stop_after_delay(session_config['round_timeout']),
               retry_error_callback=do_if_round_times_out,
               retry=retry_if_exception_type(CombinersNotDoneException))
        def combiners_done():

            round = self.statestore.get_round(round_id)
            if 'combiners' not in round:
                # TODO: use logger
                print("CONTROL: Waiting for combiners to update model...", flush=True)
                raise CombinersNotDoneException("Combiners have not yet reported.")

            if len(round['combiners']) < len(participating_combiners):
                print("CONTROL: Waiting for combiners to update model...", flush=True)
                raise CombinersNotDoneException("All combiners have not yet reported.")

            return True

        combiners_done()

        # Due to the distributed nature of the computation, there might be a
        # delay before combiners have reported the round data to the db,
        # so we need some robustness here.
        @retry(wait=wait_random(min=0.1, max=1.0),
               retry=retry_if_exception_type(KeyError))
        def check_combiners_done_reporting():
            round = self.statestore.get_round(round_id)
            combiners = round['combiners']
            return combiners

        _ = check_combiners_done_reporting()

        round = self.statestore.get_round(round_id)
        round_valid = self.evaluate_round_validity_policy(round)
        if not round_valid:
            print("REDUCER CONTROL: Round invalid!", flush=True)
            self.set_round_status(round_id, 'Failed')
            return None, self.statestore.get_round(round_id)

        print("CONTROL: Reducing combiner level models...", flush=True)
        # Reduce combiner models into a new global model
        round_data = {}
        try:
            round = self.statestore.get_round(round_id)
            model, data = self.reduce(round['combiners'])
            round_data['reduce'] = data
            print("CONTROL: Done reducing models from combiners!", flush=True)
        except Exception as e:
            print("CONTROL: Failed to reduce models from combiners: {}".format(
                e), flush=True)
            self.set_round_status(round_id, 'Failed')
            return None, self.statestore.get_round(round_id)

        # Commit the new global model to the model trail
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
            self.set_round_status(round_id, 'Failed')
            return None, self.statestore.get_round(round_id)

        # Ask combiners to validate the new global model
        validate = session_config['validate']
        if validate:
            combiner_config = copy.deepcopy(session_config)
            combiner_config['round_id'] = round_id
            combiner_config['model_id'] = self.statestore.get_latest_model()
            combiner_config['task'] = 'validation'
            combiner_config['helper_type'] = self.statestore.get_helper()

            validating_combiners = self.get_participating_combiners(
                combiner_config)

            for combiner, combiner_config in validating_combiners:
                try:
                    print("CONTROL: Submitting validation round to combiner {}".format(
                        combiner), flush=True)
                    combiner.submit(combiner_config)
                except CombinerUnavailableError:
                    self._handle_unavailable_combiner(combiner)
                    pass

        self.set_round_data(round_id, round_data)
        self.set_round_status(round_id, 'Finished')
        return model_id, self.statestore.get_round(round_id)

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

        for combiner in combiners:
            name = combiner['name']
            model_id = combiner['model_id']
            # TODO: Handle inactive RPC error in get_model and raise specific error
            print("REDUCER: Fetching model ({model_id}) from combiner {name}".format(
                model_id=model_id, name=name), flush=True)
            try:
                tic = time.time()
                combiner_interface = self.get_combiner(name)
                data = combiner_interface.get_model(model_id)
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
        if not self.statestore.latest_model():
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
        combiner_config['model_id'] = self.statestore.get_latest_model()
        combiner_config['task'] = 'inference'
        combiner_config['helper_type'] = self.statestore.get_framework()

        # Select combiners
        validating_combiners = self.get_participating_combiners(
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
