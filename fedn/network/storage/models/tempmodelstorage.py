import os
from io import BytesIO

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger

CHUNK_SIZE = 1024 * 1024


class TempModelStorage:
    """Class for managing local temporary models on file on combiners."""

    def __init__(self):
        self.default_dir = os.environ.get("FEDN_MODEL_DIR", "/tmp/models")  # set default to tmp
        if not os.path.exists(self.default_dir):
            os.makedirs(self.default_dir)

        self.models = {}
        self.model_latest_id = ""

    def exist(self, model_id):
        if model_id in self.models.keys():
            return True
        return False
    
    def set_latest_id(self, model_id):
        """Set the latest model id."""
        if model_id in self.models:
            self.model_latest_id = model_id
            logger.info(f"TEMPMODELSTORAGE: Set latest model id to {model_id}")
        else:
            logger.error(f"TEMPMODELSTORAGE: Model id {model_id} does not exist in models.")

    def get(self, model_id):
        try:
            if self.models[model_id]["state"] != fedn.ModelStatus.OK:
                logger.warning("File not ready! Try again")
                return None
        except KeyError:
            logger.error("TEMPMODELSTORAGE: model_id {} does not exist".format(model_id))
            return None

        obj = BytesIO()
        with open(os.path.join(self.default_dir, str(model_id)), "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                obj.write(chunk)

        obj.seek(0, 0)
        return obj

    def get_file_hdl(self, model_id):
        """Returns a handle to a potential new model file.

        User is responsible for closing the file.
        :param model_id:
        :return: handle to the model file
        :rtype: file
        """
        try:
            filename = self.models[model_id]["filename"]
        except KeyError:
            filename = os.path.join(self.default_dir, str(model_id))
            self.models[model_id] = {"filename": filename, "state": fedn.ModelStatus.IN_PROGRESS}
        return open(filename, "wb")

    def set_model(self, model_id: str, model_stream: BytesIO, checksum: str = None):
        """Set model in temp storage.

        :param model_id: The id of the model.
        :type model_id: str
        :param model_stream: The model stream.
        :type model_stream: BytesIO
        """
        model_stream.seek(0, 0)
        with self.get_file_hdl(model_id) as f:
            while True:
                chunk = model_stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
        if self.finalize(model_id, checksum):
            logger.info("TEMPMODELSTORAGE: Model {} added.".format(model_id))
        else:
            logger.error("TEMPMODELSTORAGE: Model {} failed.".format(model_id))
            return False

    def finalize(self, model_id, checksum):
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
        
    def is_latest_ready(self):
        """Check if the latest model is ready.

        :return: True if the latest model is ready, else False.
        :rtype: bool
        """
        try:
            return self.models[self.model_latest_id]["state"] == fedn.ModelStatus.OK, self.model_latest_id
        except KeyError:
            logger.error("TEMPMODELSTORAGE: Latest model_id {} is not ready".format(self.model_latest_id))
            return False, None

    def get_checksum(self, model_id):
        try:
            checksum = self.models[model_id]["checksum"]
        except KeyError:
            logger.error("TEMPMODELSTORAGE: model_id {} does not exist".format(model_id))
            return None
        return checksum

    # Delete model from disk
    def delete(self, model_id):
        try:
            os.remove(os.path.join(self.default_dir, str(model_id)))
            logger.info("TEMPMODELSTORAGE: Deleted model with id: {}".format(model_id))
            # Delete id from metadata and models dict
            del self.models[model_id]
        except FileNotFoundError:
            logger.error("TEMPMODELSTORAGE: Could not delete model {} from disk. File not found!".format(model_id))
            return False
        return True

    # Delete all models from disk
    def delete_all(self):
        model_ids = list(self.models.keys())
        for model_id in model_ids:
            self.delete(model_id)
        return True
