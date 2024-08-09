import os
from io import BytesIO

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.log_config import logger
from fedn.network.storage.models.modelstorage import ModelStorage

CHUNK_SIZE = 1024 * 1024


class TempModelStorage(ModelStorage):
    """Class for managing local temporary models on file on combiners."""

    def __init__(self):
        self.default_dir = os.environ.get("FEDN_MODEL_DIR", "/tmp/models")  # set default to tmp
        if not os.path.exists(self.default_dir):
            os.makedirs(self.default_dir)

        self.models = {}
        self.models_metadata = {}

    def exist(self, model_id):
        if model_id in self.models.keys():
            return True
        return False

    def get(self, model_id):
        try:
            if self.models_metadata[model_id] != fedn.ModelStatus.OK:
                logger.warning("File not ready! Try again")
                return None
        except KeyError:
            logger.error("No such model has been made available yet!")
            return None

        obj = BytesIO()
        with open(os.path.join(self.default_dir, str(model_id)), "rb") as f:
            obj.write(f.read())

        obj.seek(0, 0)
        return obj

    def get_ptr(self, model_id):
        """:param model_id:
        :return:
        """
        try:
            f = self.models[model_id]["file"]
        except KeyError:
            f = open(os.path.join(self.default_dir, str(model_id)), "wb")

        self.models[model_id] = {"file": f}
        return self.models[model_id]["file"]

    def get_model_metadata(self, model_id):
        try:
            status = self.models_metadata[model_id]
        except KeyError:
            status = fedn.ModelStatus.UNKNOWN
        return status

    def set_model_metadata(self, model_id, model_metadata):
        self.models_metadata.update({model_id: model_metadata})

    # Delete model from disk
    def delete(self, model_id):
        try:
            os.remove(os.path.join(self.default_dir, str(model_id)))
            logger.info("TEMPMODELSTORAGE: Deleted model with id: {}".format(model_id))
            # Delete id from metadata and models dict
            del self.models_metadata[model_id]
            del self.models[model_id]
        except FileNotFoundError:
            logger.error("Could not delete model from disk. File not found!")
            return False
        return True

    # Delete all models from disk
    def delete_all(self):
        ids_pop = []
        for model_id in self.models.keys():
            try:
                os.remove(os.path.join(self.default_dir, str(model_id)))
                logger.info("TEMPMODELSTORAGE: Deleted model with id: {}".format(model_id))
                # Add id to list of ids to pop/delete from metadata and models dict
                ids_pop.append(model_id)
            except FileNotFoundError:
                logger.error("TEMPMODELSTORAGE: Could not delete model {} from disk. File not found!".format(model_id))
        # Remove id from metadata and models dict
        for model_id in ids_pop:
            del self.models_metadata[model_id]
            del self.models[model_id]
        return True
