import os
from io import BytesIO

import fedn.common.net.grpc.fedn_pb2 as fedn
from fedn.common.storage.models.modelstorage import ModelStorage

CHUNK_SIZE = 1024 * 1024


class TempModelStorage(ModelStorage):
    """ Class for model storage using temporary files on local disk.

    Models are stored as files on disk. 

    """

    def __init__(self):

        self.default_dir = os.environ.get(
            'FEDN_MODEL_DIR', '/tmp/models')  # set default to tmp
        if not os.path.exists(self.default_dir):
            os.makedirs(self.default_dir)

        # TODO should read in already existing temp models if crashed? or fetch new on demand (default)
        self.models = {}
        self.models_metadata = {}

    def exist(self, model_id):
        """

        :param model_id:
        :type model_id: str
        :return: True if exists in storage, otherwise False.
        """
        if model_id in self.models.keys():
            return True
        return False

    def get(self, model_id):
        """

        :param model_id:
        :return:
        """
        try:
            if self.models_metadata[model_id] != fedn.ModelStatus.OK:
                print("File not ready! Try again", flush=True)
                return None
        except KeyError:
            print("No such model have been made available yet!", flush=True)
            return None

        obj = BytesIO()
        with open(os.path.join(self.default_dir, str(model_id)), 'rb') as f:
            obj.write(f.read())

        obj.seek(0, 0)
        return obj

    def get_ptr(self, model_id):
        """

        :param model_id:
        :return:
        """
        try:
            f = self.models[model_id]['file']
        except KeyError:
            f = open(os.path.join(self.default_dir, str(model_id)), "wb")

        self.models[model_id] = {'file': f}
        return self.models[model_id]['file']

    def get_meta(self, model_id):
        """

        :param model_id:
        :return:
        """
        return self.models_metadata[model_id]

    def set_meta(self, model_id, model_metadata):
        """

        :param model_id:
        :param model_metadata:
        """
        self.models_metadata.update({model_id: model_metadata})

    # Delete model from disk
    def delete(self, model_id):
        """ Delete model from temp disk/storage

        :param model_id: model id
        :type model_id: str
        :return: True if successful, False otherwise
        :rtype: bool
        """
        try:
            os.remove(os.path.join(self.default_dir, str(model_id)))
            print("TEMPMODELSTORAGE: Deleted model with id: {}".format(model_id), flush=True)
            # Delete id from metadata and models dict
            del self.models_metadata[model_id]
            del self.models[model_id]
        except FileNotFoundError:
            print("Could not delete model from disk. File not found!", flush=True)
            return False
        return True

    # Delete all models from disk
    def delete_all(self):
        """ Delete all models from temp disk/storage

        :return: True if successful, False otherwise
        :rtype: bool
        """
        ids_pop = []
        for model_id in self.models.keys():
            try:
                os.remove(os.path.join(self.default_dir, str(model_id)))
                print("TEMPMODELSTORAGE: Deleted model with id: {}".format(model_id), flush=True)
                # Add id to list of ids to pop/delete from metadata and models dict
                ids_pop.append(model_id)
            except FileNotFoundError:
                print("TEMPMODELSTORAGE: Could not delete model {} from disk. File not found!".format(model_id), flush=True)
        # Remove id from metadata and models dict
        for model_id in ids_pop:
            del self.models_metadata[model_id]
            del self.models[model_id]
        return True
