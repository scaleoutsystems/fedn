import os
import uuid
from abc import ABC, abstractmethod
from time import sleep

import fedn.utils.helpers
from fedn.common.storage.s3.s3repo import S3ModelRepository
from fedn.common.tracer.mongotracer import MongoTracer
from fedn.network.api.network import Network
from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.state import ReducerState


class UnsupportedStorageBackend(Exception):
    pass


class MisconfiguredStorageBackend(Exception):
    pass


class MisconfiguredHelper(Exception):
    pass


class ControlBase(ABC):
    """ Base class and interface for a global controller.
        Override this class to implement a global training strategy (control).

    :param statestore: The statestore object.
    :type statestore: :class:`fedn.network.statestore.statestorebase.StateStoreBase`
    """

    @abstractmethod
    def __init__(self, statestore):
        """ Constructor. """
        self._state = ReducerState.setup

        self.statestore = statestore
        if self.statestore.is_inited():
            self.network = Network(self, statestore)

        try:
            not_ready = True
            while not_ready:
                storage_config = self.statestore.get_storage_backend()
                if storage_config:
                    not_ready = False
                else:
                    sleep(5)
        except Exception:
            print(
                "REDUCER CONTROL: Failed to retrive storage configuration, exiting.", flush=True)
            raise MisconfiguredStorageBackend()

        if storage_config['storage_type'] == 'S3':
            self.model_repository = S3ModelRepository(storage_config['storage_config'])
        else:
            print("REDUCER CONTROL: Unsupported storage backend, exiting.", flush=True)
            raise UnsupportedStorageBackend()

        # The tracer is a helper that manages state in the database backend
        statestore_config = statestore.get_config()
        self.tracer = MongoTracer(
            statestore_config['mongo_config'], statestore_config['network_id'])

        if self.statestore.is_inited():
            self._state = ReducerState.idle

    @abstractmethod
    def session(self, config):
        pass

    @abstractmethod
    def round(self, config, round_number):
        pass

    @abstractmethod
    def reduce(self, combiners):
        pass

    def get_helper(self):
        """ Get a helper instance from global config.

        :return: Helper instance.
        :rtype: :class:`fedn.utils.plugins.helperbase.HelperBase`
        """
        helper_type = self.statestore.get_helper()
        helper = fedn.utils.helpers.get_helper(helper_type)
        if not helper:
            raise MisconfiguredHelper("Unsupported helper type {}, please configure compute_package.helper !".format(helper_type))
        return helper

    def get_state(self):
        """ Get the current state of the controller.

        :return: The current state.
        :rtype: :class:`fedn.network.state.ReducerState`
        """
        return self._state

    def idle(self):
        """ Check if the controller is idle.

        :return: True if idle, False otherwise.
        :rtype: bool
        """
        if self._state == ReducerState.idle:
            return True
        else:
            return False

    def get_model_info(self):
        """

        :return:
        """
        return self.statestore.get_model_trail()

    # TODO: remove use statestore.get_events() instead
    def get_events(self):
        """

        :return:
        """
        return self.statestore.get_events()

    def get_latest_round_id(self):
        last_round = self.statestore.get_latest_round()
        if not last_round:
            return 0
        else:
            return last_round['round_id']

    def get_latest_round(self):
        round = self.statestore.get_latest_round()
        return round

    def get_compute_package_name(self):
        """

        :return:
        """
        definition = self.statestore.get_compute_package()
        if definition:
            try:
                package_name = definition['filename']
                return package_name
            except (IndexError, KeyError):
                print(
                    "No context filename set for compute context definition", flush=True)
                return None
        else:
            return None

    def set_compute_package(self, filename, path):
        """ Persist the configuration for the compute package. """
        self.model_repository.set_compute_package(filename, path)
        self.statestore.set_compute_package(filename)

    def get_compute_package(self, compute_package=''):
        """

        :param compute_package:
        :return:
        """
        if compute_package == '':
            compute_package = self.get_compute_package_name()
        if compute_package:
            return self.model_repository.get_compute_package(compute_package)
        else:
            return None

    def new_session(self, config):
        """ Initialize a new session in backend db. """

        if "session_id" not in config.keys():
            session_id = uuid.uuid4()
            config['session_id'] = str(session_id)
        else:
            session_id = config['session_id']

        self.tracer.new_session(id=session_id)
        self.tracer.set_session_config(session_id, config)

    def new_round(self, round_data):
        """Initialize a new round in backend db. """

        self.tracer.new_round(round_data)

    def set_round_data(self, round_id, round_data):
        """ Upate round in backend db. """
        self.tracer.set_round_data(round_id, round_data)

    def set_round_status(self, round_id, status):
        """ Upate round in backend db. """
        self.tracer.set_round_status(round_id, status)

    def set_round_config(self, round_id, round_config):
        """ Upate round in backend db. """
        self.tracer.set_round_config(round_id, round_config)

    def request_model_updates(self, combiners):
        """Ask Combiner server to produce a model update. """

        cl = []
        for combiner, combiner_round_config in combiners:
            response = combiner.submit(combiner_round_config)
            cl.append((combiner, response))
        return cl

    def commit(self, model_id, model=None):
        """ Commit a model to the global model trail. The model commited becomes the lastest consensus model. """

        helper = self.get_helper()
        if model is not None:
            print("CONTROL: Saving model file temporarily to disk...", flush=True)
            outfile_name = helper.save(model)
            print("CONTROL: Uploading model to Minio...", flush=True)
            model_id = self.model_repository.set_model(
                outfile_name, is_file=True)

            print("CONTROL: Deleting temporary model file...", flush=True)
            os.unlink(outfile_name)

        print("CONTROL: Committing model {} to global model trail in statestore...".format(
            model_id), flush=True)
        self.statestore.set_latest_model(model_id)

    def get_combiner(self, name):
        for combiner in self.network.get_combiners():
            if combiner.name == name:
                return combiner
        return None

    def get_participating_combiners(self, combiner_round_config):
        """Assemble a list of combiners able to participate in a round as
           descibed by combiner_round_config.
        """
        combiners = []
        for combiner in self.network.get_combiners():
            try:
                combiner_state = combiner.report()
            except CombinerUnavailableError:
                self._handle_unavailable_combiner(combiner)
                combiner_state = None

            if combiner_state is not None:
                is_participating = self.evaluate_round_participation_policy(
                    combiner_round_config, combiner_state)
                if is_participating:
                    combiners.append((combiner, combiner_round_config))
        return combiners

    def evaluate_round_participation_policy(self, compute_plan, combiner_state):
        """ Evaluate policy for combiner round-participation.
            A combiner participates if it is responsive and reports enough
            active clients to participate in the round.
        """

        if compute_plan['task'] == 'training':
            nr_active_clients = int(combiner_state['nr_active_trainers'])
        elif compute_plan['task'] == 'validation':
            nr_active_clients = int(combiner_state['nr_active_validators'])
        else:
            print("Invalid task type!", flush=True)
            return False

        if int(compute_plan['clients_required']) <= nr_active_clients:
            return True
        else:
            return False

    def evaluate_round_start_policy(self, combiners):
        """ Check if the policy to start a round is met. """
        if len(combiners) > 0:

            return True
        else:
            return False

    def evaluate_round_validity_policy(self, round):
        """ Check if the round is valid.

            At the end of the round, before committing a model to the global model trail,
            we check if the round validity policy has been met. This can involve
            e.g. asserting that a certain number of combiners have reported in an
            updated model, or that criteria on model performance have been met.
        """
        model_ids = []
        for combiner in round['combiners']:
            try:
                model_ids.append(combiner['model_id'])
            except KeyError:
                pass

        if len(model_ids) == 0:
            return False

        return True

    def state(self):
        """

        :return:
        """
        return self._state
