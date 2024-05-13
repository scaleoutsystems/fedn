import io
from collections import defaultdict
from io import BytesIO

from fedn.network.storage.models.modelstorage import ModelStorage

CHUNK_SIZE = 1024 * 1024


class MemoryModelStorage(ModelStorage):
    """Class for in-memory storage of model artifacts.

    Models are stored as BytesIO objects in a dictionary.

    """

    def __init__(self):
        self.models = defaultdict(io.BytesIO)
        self.models_metadata = {}

    def exist(self, model_id):
        if model_id in self.models.keys():
            return True
        return False

    def get(self, model_id):
        obj = self.models[model_id]
        obj.seek(0, 0)
        # Have to copy object to not mix up the file pointers when sending... fix in better way.
        obj = BytesIO(obj.read())
        return obj

    def get_ptr(self, model_id):
        """:param model_id:
        :return:
        """
        return self.models[model_id]

    def get_model_metadata(self, model_id):
        return self.models_metadata[model_id]

    def set_model_metadata(self, model_id, model_metadata):
        self.models_metadata.update({model_id: model_metadata})
