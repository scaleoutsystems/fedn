import copy
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

import fedn.utils.helpers
from fedn.common.storage.s3.s3repo import S3ModelRepository
from fedn.common.tracer.mongotracer import MongoTracer
from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.network import Network
from fedn.network.state import ReducerState


class UnsupportedStorageBackend(Exception):
    pass


class MisconfiguredStorageBackend(Exception):
    pass


class ControlBase(ABC):
    """ Abstract class defining helpers. """

    @abstractmethod
    def __init__(self, statestore):
        """ """
        self._state = ReducerState.setup

        self.statestore = statestore
        if self.statestore.is_inited():
            self.network = Network(self, statestore)

        try:
            config = self.statestore.get_storage_backend()
        except Exception:
            print(
                "REDUCER CONTROL: Failed to retrive storage configuration, exiting.", flush=True)
            raise MisconfiguredStorageBackend()

        statestore_config = statestore.get_config()
        self.tracer = MongoTracer(
            statestore_config['mongo_config'], statestore_config['network_id'])

        if config['storage_type'] == 'S3':
            self.model_repository = S3ModelRepository(config['storage_config'])
        else:
            print("REDUCER CONTROL: Unsupported storage backend, exiting.", flush=True)
            raise UnsupportedStorageBackend()

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
        """

        :return:
        """
        helper_type = self.statestore.get_framework()
        helper = fedn.utils.helpers.get_helper(helper_type)
        if not helper:
            print("CONTROL: Unsupported helper type {}, please configure compute_context.helper !".format(helper_type),
                  flush=True)
            return None
        return helper

    def get_state(self):
        """

        :return:
        """
        return self._state

    def idle(self):
        """

        :return:
        """
        if self._state == ReducerState.idle:
            return True
        else:
            return False

    def get_first_model(self):
        """

        :return:
        """
        return self.statestore.get_first()

    def get_latest_model(self):
        """

        :return:
        """
        return self.statestore.get_latest()

    def get_model_info(self):
        """

        :return:
        """
        return self.statestore.get_model_info()

    def get_events(self):
        """

        :return:
        """
        return self.statestore.get_events()

    def drop_models(self):
        """

        """
        self.statestore.drop_models()

    def get_compute_context(self):
        """

        :return:
        """
        definition = self.statestore.get_compute_context()
        if definition:
            try:
                context = definition['filename']
                return context
            except (IndexError, KeyError):
                print(
                    "No context filename set for compute context definition", flush=True)
                return None
        else:
            return None

    def set_compute_context(self, filename, path):
        """ Persist the configuration for the compute package. """
        self.model_repository.set_compute_context(filename, path)
        self.statestore.set_compute_context(filename)

    def get_compute_package(self, compute_package=''):
        """

        :param compute_package:
        :return:
        """
        if compute_package == '':
            compute_package = self.get_compute_context()
        return self.model_repository.get_compute_package(compute_package)

    def commit(self, model_id, model=None):
        """ Commit a model to the global model trail. The model commited becomes the lastest consensus model. """

        helper = self.get_helper()
        if model is not None:
            print("Saving model to disk...", flush=True)
            outfile_name = helper.save_model(model)
            print("DONE", flush=True)
            print("Uploading model to Minio...", flush=True)
            model_id = self.model_repository.set_model(
                outfile_name, is_file=True)
            print("DONE", flush=True)
            os.unlink(outfile_name)

        self.statestore.set_latest(model_id)

    def set_combiner_model(self, combiners, model_id):
        """ Distribute a model to all active combiner nodes and set it as the current combiner model. """
        if not model_id:
            print("GOT NO MODEL TO SET! Have you seeded the FedML model?", flush=True)
            return
        for combiner in combiners:
            _ = combiner.set_model_id(model_id)

    def _check_combiners_out_of_sync(self, combiners=None):
        """ Check if combiners have completed model updates by
            checking if their active model_id differs from the
            controller latest model_id.
        """

        if not combiners:
            combiners = self.network.get_combiners()

        out_of_sync = []
        for combiner in combiners:
            try:
                model_id = combiner.get_model_id()
            except CombinerUnavailableError:
                self._handle_unavailable_combiner(combiner)
                model_id = None

            if model_id and (model_id != self.get_latest_model()):
                out_of_sync.append(combiner)
        return out_of_sync

    def get_participating_combiners(self, combiner_round_config):

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
        """ Evaluate reducer level policy for combiner round-participation. """

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

    def evaluate_round_validity_policy(self, combiners):
        """
            At the end of the round, before committing a model to the global model trail,
            we check if the round validity policy has been met. This can involve
            e.g. asserting that a certain number of combiners have reported in an
            updated model, or that criteria on model performance have been met.
        """
        if combiners == []:
            return False
        else:
            return True

    def _select_participating_combiners(self, compute_plan):
        participating_combiners = []
        for combiner in self.network.get_combiners():
            try:
                combiner_state = combiner.report()
            except CombinerUnavailableError:
                self._handle_unavailable_combiner(combiner)
                combiner_state = None

            if combiner_state:
                is_participating = self.evaluate_round_participation_policy(
                    compute_plan, combiner_state)
                if is_participating:
                    participating_combiners.append((combiner, compute_plan))
        return participating_combiners

    def state(self):
        """

        :return:
        """
        return self._state
