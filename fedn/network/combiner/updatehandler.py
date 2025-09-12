import json
import queue
import threading
import time
import traceback
from typing import Dict, List

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import ModelService
from fedn.utils.model import FednModel


class ModelUpdateError(Exception):
    pass


class UpdateHandler:
    """Update handler.

    Responsible for receiving, loading and supplying client model updates.

    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    """

    def __init__(self, modelservice: ModelService) -> None:
        self.modelservice = modelservice

        self.session_queue: Dict[str, SessionQueue] = {}

    def get_session_queue(self, session_id):
        """Get the session queue for the given session ID.

        If the session queue does not exist, create a new one.
        :param session_id: The session ID
        :type session_id: str
        :return: The group of model updates.
        :rtype: SessionQueue
        """
        if session_id not in self.session_queue:
            logger.info("UPDATE HANDLER: Creating new update queue for session {}".format(session_id))
            self.session_queue[session_id] = SessionQueue(self, session_id=session_id)
        return self.session_queue[session_id]

    def delete_model(self, model_update: fedn.ModelUpdate):
        self.modelservice.temp_model_storage.delete(model_update.model_update_id)
        logger.info("UPDATE HANDLER: Deleted model update {} from storage.".format(model_update.model_update_id))

    def next_model_update(self, session_id):
        """Get the next model update from the queue.

        :param session_id: The session ID
        :type session_id: str
        :return: The model update.
        :rtype: fedn.network.grpc.fedn.proto.ModelUpdate
        :raises: queue.Empty
        """
        if session_id in self.session_queue:
            return self.session_queue[session_id].next_model_update()
        else:
            raise RuntimeError("No update queue set. Please create an update queue before calling this method.")

    def on_model_update(self, model_update: fedn.ModelUpdate):
        """Callback when a new client model update is recieved.

        Performs (optional) validation and pre-processing,
        and then puts the update id on the aggregation queue.
        Override in subclass as needed.

        :param model_update: fedn.network.grpc.fedn.proto.ModelUpdate
        :type model_id: str
        """
        try:
            logger.info("UPDATE HANDLER: callback received model update {}".format(model_update.model_update_id))

            # Validate the update and metadata
            valid_update = self._validate_model_update(model_update)
            if valid_update:
                # Push the model update to the processing queue
                if model_update.session_id in self.session_queue:
                    self.session_queue[model_update.session_id].add_model_update(model_update)
                else:
                    logger.warning("UPDATE HANDLER: No session queue found for session {}, skipping.".format(model_update.session_id))
            else:
                logger.warning("UPDATE HANDLER: Invalid model update, skipping.")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("UPDATE HANDLER: failed to receive model update: {}".format(e))
            logger.error(tb)
            pass

    def _validate_model_update(self, model_update: fedn.ModelUpdate):
        """Validate the model update.

        :param model_update: A ModelUpdate message.
        :type model_update: object
        :return: True if the model update is valid, False otherwise.
        :rtype: bool
        """
        try:
            data = json.loads(model_update.meta)["training_metadata"]
            _ = data["num_examples"]
        except KeyError:
            tb = traceback.format_exc()
            logger.error("UPDATE HANDLER: Invalid model update, missing metadata.")
            logger.error(tb)
            return False

        if not self.modelservice.exist(model_update.model_update_id):
            logger.error("UPDATE HANDLER: Model update {} not found.".format(model_update.model_update_id))
            return False

        return True

    def load_model_update(self, model_update: fedn.ModelUpdate, helper):
        """Load the memory representation of the model update.

        Load the model update paramters and the
        associate metadata into memory.

        :param model_update: The model update.
        :type model_update: fedn.network.grpc.fedn.proto.ModelUpdate
        :param helper: A helper object.
        :type helper: fedn.utils.helpers.helperbase.Helper
        :return: A tuple of (parameters, metadata)
        :rtype: tuple
        """
        model_id = model_update.model_update_id
        model_params = self.load_model_params(helper, model_id)
        # Get relevant metadata
        metadata = json.loads(model_update.meta)
        if "config" in metadata.keys():
            # Used in Python client
            config = json.loads(metadata["config"])
        else:
            # Used in C++ client
            config = json.loads(model_update.config)
        training_metadata = metadata["training_metadata"]
        if "round_id" in config:
            training_metadata["round_id"] = config["round_id"]

        return model_params, training_metadata

    def load_model_params(self, helper, model_id):
        """Load model update with id model_id into its memory representation.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`
        :param model_id: The ID of the model update, UUID in str format
        :type model_id: str
        """
        fedn_model = self.get_model(model_id)
        if fedn_model:
            try:
                model = fedn_model.get_model_params(helper)
            except IOError:
                logger.warning("UPDATE HANDLER: Failed to load model!")
        else:
            raise ModelUpdateError("Failed to load model.")

        return model

    def get_model(self, model_id) -> FednModel:
        """Load model update object and return it as FednModel.

        :param model_id: The ID of the model
        :type model_id: str
        :param retry: number of times retrying load model update, defaults to 3
        :type retry: int, optional
        :return: Updated model
        :rtype: class: `io.BytesIO`
        """
        return self.modelservice.get_model(model_id)

    def flush_session(self, session_id):
        """Flush the session queue for the given session ID.

        :param session_id: The session ID
        :type session_id: str
        """
        if session_id in self.session_queue:
            logger.info("UPDATE HANDLER: Flushing update queue for session {}".format(session_id))
            self.session_queue[session_id].flush_session()


class BackwardHandler:
    """Backward handler.

    Handles the backward completion messages during split learning backward passes.

    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    """

    def __init__(self) -> None:
        self.backward_completions = queue.Queue()

    def waitforbackwardcompletion(self, config, required_backward_completions=-1, polling_interval=0.1):
        """Wait for backward completion messages.

        :param config: The round config object
        :param required_backward_completions: Number of required backward completions
        """
        time_window = float(config["round_timeout"])
        tt = 0.0

        while tt < time_window:
            if self.backward_completions.qsize() >= required_backward_completions:
                break

            time.sleep(polling_interval)
            tt += polling_interval

    def clear_backward_completions(self):
        """Clear the backward completions queue."""
        while not self.backward_completions.empty():
            try:
                self.backward_completions.get_nowait()
            except queue.Empty:
                break


class SessionQueue:
    def __init__(
        self,
        update_handler: UpdateHandler,
        session_id: str,
        accept_stragglers: bool = False,
    ):
        self.session_id = session_id
        self.round_id: str = None
        self.update_handler = update_handler

        self.model_update: queue.Queue[fedn.ModelUpdate] = queue.Queue()
        self.model_update_stragglers: queue.Queue[fedn.ModelUpdate] = queue.Queue()

        self.expected_correlation_ids = []
        self.straggler_correlation_ids: List[str] = []

        self._accept_stragglers = accept_stragglers

        self.lock = threading.RLock()

    def add_model_update(self, model_update: fedn.ModelUpdate) -> bool:
        if model_update.session_id != self.session_id:
            # This indicates an error in the implementation
            logger.error(f"UPDATE HANDLER: Model update {model_update.model_update_id} is ignored due to wrong session id.")
            self.handle_invalid_model_update(model_update)
            return False

        with self.lock:
            if model_update.correlation_id in self.expected_correlation_ids:
                # Expected model update
                self.expected_correlation_ids.remove(model_update.correlation_id)
                self.model_update.put(model_update)
                return True
            elif model_update.correlation_id in self.straggler_correlation_ids:
                # Straggler model update
                self.straggler_correlation_ids.remove(model_update.correlation_id)
                if self._accept_stragglers:
                    self.model_update_stragglers.put(model_update)
                    return True
                else:
                    logger.warning(f"UPDATE HANDLER: Model update {model_update.model_update_id} is ignored due to late arrival.")
                    self.handle_ignored_model_update(model_update)
            else:
                # Unknown model update
                logger.error(f"UPDATE HANDLER: Model update {model_update.model_update_id} is ignored due to invalid correlation id.")
                self.handle_invalid_model_update(model_update)
        return False

    def get_all_outstanding_correlation_ids(self) -> List[str]:
        """Get all outstanding correlation IDs.

        :return: List of outstanding correlation IDs.
        :rtype: List[str]
        """
        with self.lock:
            return self.expected_correlation_ids + self.straggler_correlation_ids

    def handle_invalid_model_update(self, model_update: fedn.ModelUpdate):
        """Handle invalid model update.

        :param model_update: The model update.
        :type model_update: fedn.network.grpc.fedn.proto.ModelUpdate
        """
        # TODO: Maybe want to properly track invalid model updates somehow
        # For now, just delete them
        self.update_handler.delete_model(model_update)

    def handle_ignored_model_update(self, model_update: fedn.ModelUpdate):
        """Handle invalid model update.

        :param model_update: The model update.
        :type model_update: fedn.network.grpc.fedn.proto.ModelUpdate
        """
        # TODO: Maybe want to properly track ignored model updates somehow
        # For now, just delete them
        self.update_handler.delete_model(model_update)

    def start_round_queue(self, round_id, expected_correlation_ids: List[str], accept_stragglers: bool = False):
        """Progress to the next round transfering stragglers to the next round."""
        with self.lock:
            self.round_id = round_id
            self._accept_stragglers = accept_stragglers

            # Transfer stragglers to the next round
            self.straggler_correlation_ids.extend(self.expected_correlation_ids)
            self.expected_correlation_ids = expected_correlation_ids

            # Transfer model updates to the next round
            # Model updates might contain some stragglers that was sent after the round
            # was finished, so we need to transfer them to the next round
            while not self.model_update.empty():
                model_update = self.model_update.get()
                if self._accept_stragglers:
                    self.model_update_stragglers.put(model_update)
                else:
                    logger.warning(f"UPDATE HANDLER: Model update {model_update.model_update_id} is ignored due to late arrival.")
                    self.handle_ignored_model_update(model_update)

    def finish_session(self):
        """Finish the session"""
        with self.lock:
            self.expected_correlation_ids = []
            self.straggler_correlation_ids = []
            while not self.model_update.empty():
                model_update = self.model_update.get()
                logger.warning(f"UPDATE HANDLER: Model update {model_update.model_update_id} is ignored due to session end.")
                self.handle_ignored_model_update(model_update)
            while not self.model_update_stragglers.empty():
                model_update = self.model_update_stragglers.get()
                logger.warning(f"UPDATE HANDLER: Model update {model_update.model_update_id} is ignored due to session end.")
                self.handle_ignored_model_update(model_update)

    def flush_session(self):
        """Flush the session queue."""
        with self.lock:
            self.expected_correlation_ids = []
            self.straggler_correlation_ids = []
            while not self.model_update.empty():
                model_update = self.model_update.get()
                self.handle_ignored_model_update(model_update)
            while not self.model_update_stragglers.empty():
                model_update = self.model_update_stragglers.get()
                self.handle_ignored_model_update(model_update)

    def next_model_update(self):
        """Get the next model update from the queue.

        :return: The model update.
        :rtype: fedn.network.grpc.fedn.proto.ModelUpdate
        """
        try:
            return self.model_update.get(block=False)
        except queue.Empty:
            if self._accept_stragglers:
                return self.model_update_stragglers.get(block=False)
            else:
                raise queue.Empty

    def aggregation_condition(self, buffer_size=100):
        """Check if the round has enough updates to continue to aggregate.

        :param buffer_size: The number of model updates to wait for before starting aggregation, defaults to 100
        :type buffer_size: int, optional
        :return: True if the round is complete, False otherwise.
        :rtype: bool
        """
        return self.model_update.qsize() >= buffer_size
