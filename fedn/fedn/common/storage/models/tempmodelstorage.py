import fedn.common.net.grpc.fedn_pb2 as fedn
from fedn.common.storage.models.modelstorage import ModelStorage

CHUNK_SIZE = 1024 * 1024

import os


class TempModelStorage(ModelStorage):

    def __init__(self):

        self.default_dir = os.environ.get('FEDN_MODEL_DIR', '/tmp/models')  # set default to tmp
        if not os.path.exists(self.default_dir):
            os.makedirs(self.default_dir)

        # TODO should read in already existing temp models if crashed? or fetch new on demand (default)

        # self.models = defaultdict(io.BytesIO)
        self.models = {}
        self.models_metadata = {}

    def exist(self, model_id):
        if model_id in self.models.keys():
            return True
        return False

    def get(self, model_id):
        try:
            if self.models_metadata[model_id] != fedn.ModelStatus.OK:
                print("File not ready! Try again", flush=True)
                return None
        except KeyError:
            print("No such model have been made available yet!", flush=True)
            return None

        from io import BytesIO
        obj = BytesIO()
        with open(os.path.join(self.default_dir, str(model_id)), 'rb') as f:
            obj.write(f.read())

        obj.seek(0, 0)
        return obj

    def get_ptr(self, model_id):
        try:
            f = self.models[model_id]['file']
        except KeyError:
            f = open(os.path.join(self.default_dir, str(model_id)), "wb")

        self.models[model_id] = {'file': f}
        return self.models[model_id]['file']

    def get_meta(self, model_id):
        return self.models_metadata[model_id]

    def set_meta(self, model_id, model_metadata):

        self.models_metadata.update({model_id: model_metadata})
