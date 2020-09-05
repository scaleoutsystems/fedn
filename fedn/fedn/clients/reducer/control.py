from .state import ReducerState
import copy

class ReducerControl:

    def __init__(self):
        self.__state = ReducerState.idle
        self.combiners = []
        # TODO: Store in DB 
        self.model_id = None

    def _get_model_id(self):
        # TODO: get from DB backend
        return self.model_id

    def instruct(self, config):
        if self.__state == ReducerState.instructing:
            print("Already set in INSTRUCTING state", flush=True)
            return

        self.__state = ReducerState.instructing

        for round in range(config['rounds']): 
            
            round_model_id = self._get_model_id()
            if not round_model_id:
                round_model_id  = config['model_id']
            combiner_config = copy.deepcopy(config)
            combiner_config['model_id'] = round_model_id
            combiner_config['rounds'] = 1
            combiner_config['task'] = 'training'

            print("REDUCER: STARTING COMBINERS", flush=True)
            for combiner in self.combiners:
                print("REDUCER: STARTING {}".format(combiner.name), flush=True)
                combiner.start(combiner_config)
            print("REDUCER: STARTED {} COMBINERS".format(len(self.combiners), flush=True))

            # Aggregate models

        self.__state = ReducerState.monitoring

    def monitor(self, config=None):
        if self.__state == ReducerState.monitoring:
            print("monitoring")
        # todo connect to combiners and listen for globalmodelupdate request.
        # use the globalmodel received to start the reducer combiner method on received models to construct its own model.

    def add(self, combiner):
        if self.__state != ReducerState.idle:
            print("Reducer is not idle, cannot add additional combiner")
            return
        if self.find(combiner.name):
            return
        print("adding combiner {}".format(combiner.name), flush=True)
        self.combiners.append(combiner)

    def remove(self, combiner):
        if self.__state != ReducerState.idle:
            print("Reducer is not idle, cannot remove combiner")
            return
        self.combiners.remove(combiner)

    def find(self, name):
        for combiner in self.combiners:
            if name == combiner.name:
                return combiner
        return None

    def find_available_combiner(self):
        for combiner in self.combiners:
            if combiner.allowing_clients():
                return combiner
        return None

    def state(self):
        return self.__state
