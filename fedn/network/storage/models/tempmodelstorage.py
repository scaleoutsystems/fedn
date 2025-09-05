import threading
import time
from io import BytesIO

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger
from fedn.utils.model import FednModel

CHUNK_SIZE = 1024 * 1024


class TempModelStorage:
    """Class for managing local temporary models on file on combiners.

    This class provides methods to store, retrieve, and manage models in a temporary directory.
    Cached models are kept for one hour after they were last accessed.
    Manually added models will be kept until they are deleted explicitly.
    """

    def __init__(self):
        self.models = {}
        self.access_lock = threading.RLock()

    def exist(self, model_id):
        if model_id in self.models.keys():
            return True
        return False

    def get(self, model_id):
        with self.access_lock:
            if not self.exist(model_id):
                logger.error("TEMPMODELSTORAGE: model_id {} does not exist".format(model_id))
                return None
            if self.models[model_id]["state"] != fedn.ModelStatus.OK:
                logger.warning("File not ready! Try again")
                return None
            self.models[model_id]["accessed_at"] = time.time()
            return self.models[model_id]["model"]

    def _make_entry(self, model_id, model):
        """Returns a handle to a new model file.


        User is responsible for closing the file.
        :param model_id:
        :return: handle to the model file
        :rtype: file
        """
        with self.access_lock:
            now = time.time()
            if model_id in self.models:
                raise ValueError("Model with id {} already exists.".format(model_id))
            else:
                self.models[model_id] = {
                    "model": model,
                    "state": fedn.ModelStatus.IN_PROGRESS,
                    "auto_managed": False,
                    "accessed_at": now,
                }

            self._invalidate_old_models()
            return model

    def _set_model(self, model_id: str, model_lambda, checksum: str = None, auto_managed: bool = False):
        if model_id in self.models:
            raise ValueError("Model with id {} already exists.".format(model_id))
        with self.access_lock:
            try:
                self._make_entry(model_id, None)
            except Exception as e:
                logger.error("TEMPMODELSTORAGE: Error writing model {} to disk: {}".format(model_id, e))
                return False

        # Create the model using the provided lambda function
        # Do this outside the lock to avoid blocking other threads
        model = model_lambda()

        with self.access_lock:
            self.models[model_id]["model"] = model
            if self._finalize(model_id, checksum):
                self.models[model_id]["auto_managed"] = auto_managed
                logger.info("TEMPMODELSTORAGE: Model {} added.".format(model_id))
                return True
            else:
                logger.error("TEMPMODELSTORAGE: Model {} failed.".format(model_id))
            return False

    def set_model(self, model_id: str, model: FednModel, checksum: str = None, auto_managed: bool = False):
        """Set model in temp storage.

        :param model_id: The id of the model.
        :type model_id: str
        :param model: The model object.
        :type model: FednModel
        """
        return self._set_model(model_id, lambda: model, checksum, auto_managed)

    def set_model_from_stream(self, model_id: str, model_stream: BytesIO, checksum: str = None, auto_managed: bool = False):
        """Set model in temp storage.

        :param model_id: The id of the model.
        :type model_id: str
        :param model_stream: The model stream.
        :type model_stream: BytesIO
        """
        return self._set_model(model_id, lambda: FednModel.from_stream(model_stream), checksum, auto_managed)

    def set_model_with_generator(self, model_id: str, chunk_generator, checksum: str = None, auto_managed: bool = False):
        """Set model in temp storage using a generator.

        :param model_id: The id of the model.
        :type model_id: str
        :param chunk_generator: A generator that yields chunks of the model.
        :type chunk_generator: Generator[bytes, None, None]
        """
        return self._set_model(model_id, lambda: FednModel.from_chunk_generator(chunk_generator), checksum, auto_managed)

    def _finalize(self, model_id, checksum):
        """Commit the model to disk.

        :param model_id: The id of the model.
        :type model_id: str
        :param checksum: The checksum of the model.
        :type checksum: str
        """
        model: FednModel = self.models[model_id]["model"]
        if not model.verify_checksum(checksum):
            logger.error("TEMPMODELSTORAGE: Checksum failed! File is corrupted!")
            self.delete(model_id)
            return False
        self.models[model_id]["state"] = fedn.ModelStatus.OK
        return True

    def is_ready(self, model_id):
        """Check if model is ready.

        :param model_id: The id of the model.
        :type model_id: str
        :return: True if model is ready, else False.
        :rtype: bool
        """
        try:
            return self.models[model_id]["state"] == fedn.ModelStatus.OK
        except KeyError:
            logger.error("TEMPMODELSTORAGE: model_id {} does not exist".format(model_id))
            return False

    def get_checksum(self, model_id):
        try:
            model: FednModel = self.models[model_id]["model"]
        except KeyError:
            logger.error("TEMPMODELSTORAGE: model_id {} does not exist".format(model_id))
            return None
        with self.access_lock:
            return model.checksum

    def _invalidate_old_models(self):
        """Remove cached models that have not been accessed for more than 1 hours."""
        now = time.time()
        for model_id, model_info in list(self.models.items()):
            if now - model_info["accessed_at"] > 3600 and model_info["auto_managed"]:
                logger.info("TEMPMODELSTORAGE: Invalidating model {} due to inactivity.".format(model_id))
                self.delete(model_id)

    # Delete model from disk
    def delete(self, model_id):
        with self.access_lock:
            try:
                logger.info("TEMPMODELSTORAGE: Deleted model with id: {}".format(model_id))
                # Delete id from metadata and models dict
                del self.models[model_id]
            except FileNotFoundError:
                logger.error("TEMPMODELSTORAGE: Could not delete model {} from disk. File not found!".format(model_id))
                return False
            return True

    # Delete all models from disk
    def delete_all(self):
        with self.access_lock:
            self.models.clear()
        return True
