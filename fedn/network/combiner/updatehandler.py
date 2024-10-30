import json
import queue
import sys
import time
import traceback

from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import ModelService, load_model_from_bytes


class ModelUpdateError(Exception):
    pass


class UpdateHandler:
    """Update handler.

    Responsible for receiving, loading and supplying client model updates.

    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    """

    def __init__(self, modelservice: ModelService) -> None:
        self.model_updates = queue.Queue()
        self.modelservice = modelservice

        self.model_id_to_model_data = {}

    def delete_model(self, model_update):
        self.modelservice.temp_model_storage.delete(model_update.model_update_id)
        logger.info("UPDATE HANDLER: Deleted model update {} from storage.".format(model_update.model_update_id))

    def next_model_update(self):
        """Get the next model update from the queue.

        :param helper: A helper object.
        :type helper: object
        :return: The model update.
        :rtype: fedn.network.grpc.fedn.proto.ModelUpdate
        """
        model_update = self.model_updates.get(block=False)
        return model_update

    def on_model_update(self, model_update):
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
                self.model_updates.put(model_update)
            else:
                logger.warning("UPDATE HANDLER: Invalid model update, skipping.")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("UPDATE HANDLER: failed to receive model update: {}".format(e))
            logger.error(tb)
            pass

    def _validate_model_update(self, model_update):
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
        return True

    def load_model_update(self, model_update, helper):
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
        model = self.load_model(helper, model_id)
        # Get relevant metadata
        metadata = json.loads(model_update.meta)
        if "config" in metadata.keys():
            # Used in Python client
            config = json.loads(metadata["config"])
        else:
            # Used in C++ client
            config = json.loads(model_update.config)
        training_metadata = metadata["training_metadata"]
        training_metadata["round_id"] = config["round_id"]

        return model, training_metadata

    def load_model_update_byte(self, model_update):
        """Load the memory representation of the model update.

        Load the model update paramters and the
        associate metadata into memory.

        :param model_update: The model update.
        :type model_update: fedn.network.grpc.fedn.proto.ModelUpdate
        :return: A tuple of parameters(bytes), metadata
        :rtype: tuple
        """
        model_id = model_update.model_update_id
        model = self.load_model_update_bytesIO(model_id).getbuffer()
        # Get relevant metadata
        metadata = json.loads(model_update.meta)
        if "config" in metadata.keys():
            # Used in Python client
            config = json.loads(metadata["config"])
        else:
            # Used in C++ client
            config = json.loads(model_update.config)
        training_metadata = metadata["training_metadata"]
        training_metadata["round_id"] = config["round_id"]

        return model, training_metadata

    def load_model(self, helper, model_id):
        """Load model update with id model_id into its memory representation.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`
        :param model_id: The ID of the model update, UUID in str format
        :type model_id: str
        """
        model_bytesIO = self.load_model_update_bytesIO(model_id)
        if model_bytesIO:
            try:
                model = load_model_from_bytes(model_bytesIO.getbuffer(), helper)
            except IOError:
                logger.warning("UPDATE HANDLER: Failed to load model!")
        else:
            raise ModelUpdateError("Failed to load model.")

        return model

    def load_model_update_bytesIO(self, model_id, retry=3):
        """Load model update object and return it as BytesIO.

        :param model_id: The ID of the model
        :type model_id: str
        :param retry: number of times retrying load model update, defaults to 3
        :type retry: int, optional
        :return: Updated model
        :rtype: class: `io.BytesIO`
        """
        # Try reading model update from local disk/combiner memory
        model_str = self.modelservice.temp_model_storage.get(model_id)
        # And if we cannot access that, try downloading from the server
        if model_str is None:
            model_str = self.modelservice.get_model(model_id)
            # TODO: use retrying library
            tries = 0
            while tries < retry:
                tries += 1
                if not model_str or sys.getsizeof(model_str) == 80:
                    logger.warning("Model download failed. retrying")
                    time.sleep(1)
                    model_str = self.modelservice.get_model(model_id)

        return model_str

    def waitforit(self, config, buffer_size=100, polling_interval=0.1):
        """Defines the policy for how long the server should wait before starting to aggregate models.

        The policy is as follows:
            1. Wait a maximum of time_window time until the round times out.
            2. Terminate if a preset number of model updates (buffer_size) are in the queue.

        :param config: The round config object
        :type config: dict
        :param buffer_size: The number of model updates to wait for before starting aggregation, defaults to 100
        :type buffer_size: int, optional
        :param polling_interval: The polling interval, defaults to 0.1
        :type polling_interval: float, optional
        """
        time_window = float(config["round_timeout"])

        tt = 0.0
        while tt < time_window:
            if self.model_updates.qsize() >= buffer_size:
                break

            time.sleep(polling_interval)
            tt += polling_interval
