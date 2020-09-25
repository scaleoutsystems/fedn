from fedn.clients.reducer.state import ReducerStateToString, StringToReducerState
from fedn.common.storage.db.mongo import connect_to_mongodb
from .reducerstatestore import ReducerStateStore


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
        return StringToReducerState(self.state.find_one()['current_state'])

    def transition(self, state):
        old_state = self.state.find_one({'state': 'current_state'})
        if old_state != state:
            return self.state.update({'state': 'current_state'}, {'state': ReducerStateToString(state)}, True)
        else:
            print("Not updating state, already in {}".format(ReducerStateToString(state)))

    def set_latest(self, model_id):
        from datetime import datetime
        x = self.latest_model.update({'key': 'current_model'}, {'$set': {'model': model_id}}, True)
        self.models.update({'key': 'models'}, {'$push': {'model': model_id, 'committed_at': str(datetime.now())}}, True)

    def get_latest(self):
        ret = self.latest_model.find({'key': 'current_model'})
        try:
            retcheck = ret[0]['model']
            if retcheck == '' or retcheck == ' ':  #ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None
