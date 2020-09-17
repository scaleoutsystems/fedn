from abc import ABC, abstractmethod

from fedn.common.storage.db.mongo import connect_to_mongodb


class ReducerStateStore(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set(self, key, value):
        pass

    @abstractmethod
    def get(self, key):
        pass


model_info_dict = {'model_id': '',
                   'model_description': '',
                   'created_at': '',
                   'created_by': '',
                   'validated_at': '',
                   'validated_by': '',
                   'signed_by': ''
                   }


class MongoReducerStateStore(ReducerStateStore):
    def __init__(self):
        try:
            self.mdb = connect_to_mongodb()
            self.state = self.mdb['reducer_state']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.state = None
            raise

    def set(self, key, value):
        self.state.insert({'key': key, 'value': value})

    def get(self, key):
        return self.state.find({'key': key})

    def push_model(self, model_id):
        self.state.update({'key': 'models'}, {'$push': {'model': model_id}})
