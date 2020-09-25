from fedn.common.storage.models.modelstorage import ModelStorage
from collections import defaultdict
import io

CHUNK_SIZE = 1024 * 1024


class MemoryModelStorage(ModelStorage):

    def __init__(self):
        import tempfile
        # self.dir = tempfile.TemporaryDirectory()
        self.models = defaultdict(io.BytesIO)
        self.models_metadata = {}

    def exist(self, model_id):
        if model_id in self.models.keys():
            return True
        return False

    def get(self, model_id):
        from io import BytesIO
        obj = self.models[model_id]
        obj.seek(0, 0)
        # Have to copy object to not mix up the file pointers when sending... fix in better way.
        obj = BytesIO(obj.read())
        return obj

    def get_ptr(self, model_id):
        return self.models[model_id]

    def get_meta(self, model_id):
        return self.models_metadata[model_id]

    def set_meta(self, model_id, model_metadata):
        self.models_metadata.update({model_id: model_metadata})