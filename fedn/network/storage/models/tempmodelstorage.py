import os
import threading
import time
from io import BytesIO

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger

CHUNK_SIZE = 1024 * 1024


class TempModelStorage:
    """Class for managing local temporary models on file on combiners.

    This class provides methods to store, retrieve, and manage models in a temporary directory.
    Cached models are kept for one hour after they were last accessed.
    Manually added models will be kept until they are deleted explicitly.
    """

    def __init__(self):
        self.default_dir = os.environ.get("FEDN_MODEL_DIR", "/tmp/models")  # set default to tmp
        if not os.path.exists(self.default_dir):
            os.makedirs(self.default_dir)

        self.models = {}
        self.access_lock = threading.RLock()

    def _file_name(self, model_id):
        """Get the file name for a model_id."""
        return os.path.join(self.default_dir, str(model_id))

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
            return open(self.models[model_id]["filename"], "rb")

    def _get_new_file_hdl(self, model_id):
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
                filename = self._file_name(model_id)
                self.models[model_id] = {
                    "filename": filename,
                    "state": fedn.ModelStatus.IN_PROGRESS,
                    "checksum": None,
                    "auto_managed": False,
                    "accessed_at": now,
                }

            self._invalidate_old_models()
            return open(filename, "wb")

    def set_model(self, model_id: str, model_stream: BytesIO, checksum: str = None, auto_managed: bool = False):
        """Set model in temp storage.

        :param model_id: The id of the model.
        :type model_id: str
        :param model_stream: The model stream.
        :type model_stream: BytesIO
        """
        with self.access_lock:
            try:
                model_stream.seek(0, 0)
                with self._get_new_file_hdl(model_id) as f:
                    while True:
                        chunk = model_stream.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
            except Exception as e:
                logger.error("TEMPMODELSTORAGE: Error writing model {} to disk: {}".format(model_id, e))
                try:
                    os.remove(self._file_name(model_id))
                except FileNotFoundError:
                    pass
                return False

            if self._finalize(model_id, checksum):
                self.models[model_id]["auto_managed"] = auto_managed
                logger.info("TEMPMODELSTORAGE: Model {} added.".format(model_id))
                return True
            else:
                logger.error("TEMPMODELSTORAGE: Model {} failed.".format(model_id))
            return False

    def set_model_with_generator(self, model_id: str, chunk_generator, checksum: str = None, auto_managed: bool = False):
        """Set model in temp storage using a generator.

        :param model_id: The id of the model.
        :type model_id: str
        :param chunk_generator: A generator that yields chunks of the model.
        :type chunk_generator: Generator[bytes, None, None]
        """
        with self.access_lock:
            try:
                with self._get_new_file_hdl(model_id) as f:
                    for chunk in chunk_generator:
                        f.write(chunk)
            except Exception as e:
                logger.error("TEMPMODELSTORAGE: Error writing model {} to disk: {}".format(model_id, e))
                try:
                    os.remove(self._file_name(model_id))
                except FileNotFoundError:
                    pass
                return False

            if self._finalize(model_id, checksum):
                self.models[model_id]["auto_managed"] = auto_managed
                logger.info("TEMPMODELSTORAGE: Model {} added.".format(model_id))
                return True
            else:
                logger.error("TEMPMODELSTORAGE: Model {} failed.".format(model_id))
                return False

    def _finalize(self, model_id, checksum):
        """Commit the model to disk.

        :param model_id: The id of the model.
        :type model_id: str
        :param checksum: The checksum of the model.
        :type checksum: str
        """
        downloaded_file_checksum = self.compute_checksum(model_id)
        if checksum and downloaded_file_checksum != checksum:
            logger.error("TEMPMODELSTORAGE: Checksum failed! File is corrupted!")
            self.delete(model_id)
            return False
        self.models[model_id]["checksum"] = downloaded_file_checksum
        self.models[model_id]["state"] = fedn.ModelStatus.OK
        return True

    def compute_checksum(self, model_id):
        """Compute checksum for model.

        :param model_id: The id of the model.
        :type model_id: str
        :return: The checksum of the model.
        :rtype: str
        """
        try:
            logger.debug(f"TEMPMODELSTORAGE: Computing checksum for {model_id} is not implemented yet")
            # with open(os.path.join(self.default_dir, str(model_id)), "rb") as f:
            # checksum = fedn.compute_checksum(f)
            checksum = None
            return checksum
        except FileNotFoundError:
            logger.error("TEMPMODELSTORAGE: model_id {} does not exist".format(model_id))
            return None

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
            checksum = self.models[model_id]["checksum"]
        except KeyError:
            logger.error("TEMPMODELSTORAGE: model_id {} does not exist".format(model_id))
            return None
        return checksum

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
                os.remove(self._file_name(model_id))
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
            logger.info("TEMPMODELSTORAGE: Deleting all models from disk.")
            # Delete all models from disk
            for model_id in list(self.models.keys()):
                self.delete(model_id)
            # Clear the models dictionary
            self.models.clear()
        return True
