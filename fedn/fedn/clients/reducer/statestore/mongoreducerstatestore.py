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
            self.compute_context = self.mdb['compute_context']
            self.compute_context_trail = self.mdb['compute_context_trail']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.state = None
            self.models = None
            self.latest_model = None
            self.compute_context = None
            self.compute_context_trail = None
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

    def set_compute_context(self, filename):
        from datetime import datetime
        x = self.compute_context.update({'key': 'package'}, {'$set': {'filename': filename}}, True)
        self.compute_context_trail.update({'key': 'package'}, {'$push': {'filename': filename, 'committed_at': str(datetime.now())}}, True)

    def get_compute_context(self):
        ret = self.compute_context.find({'key': 'package'})
        try:
            retcheck = ret[0]['filename']
            if retcheck == '' or retcheck == ' ':  #ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None
