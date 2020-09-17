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
            self.state = self.mdb['state']
            self.models = self.mdb['models']
            self.latest_model = self.mdb['latest_model']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.state = None
            self.models = None
            self.latest_model = None
            raise

    def state(self):
        return self.state.find_one()

    def transitions(self, state):
        return self.state.insert_one({'state': state})

    def set(self, key, value):
        x = self.state.insert_one({'key': key, 'value': value})
        return x.inserted_id

    def get(self, key):
        return self.state.find({'key': key})

    def set_latest(self, model_id):
        # self.model_id = model_id
        from datetime import datetime
        x = self.latest_model.update({'key': 'current_model'}, {'$set': {'model': model_id}}, True)
        self.models.update({'key': 'models'}, {'$push': {'model': model_id, 'committed_at': str(datetime.now())}}, True)

    def get_latest(self):
        # return self.model_id
        ret = self.latest_model.find({'key': 'current_model'})
        try:
            return ret[0]['model']
        except KeyError:
            return None
