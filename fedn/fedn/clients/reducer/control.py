from .state import ReducerState
import copy


class Model:
    """ (DB) representation of a global model. """ 
    def __init__(self):

        self.model_id = None
        self.version = ""
        self.parent = ""


class ReducerControl:

    def __init__(self):
        self.__state = ReducerState.idle
        self.combiners = []

        # TODO: Use DB / Eth
        # models should be an immutable, ordered chain of global models
        self.models = []

    def get_latest_model(self):
        # TODO: get single point of thruth from DB / Eth backend
        return self.model_id

    def commit(self,model_id):
        """ Commit a model. This establishes this model as the lastest consensus model. """

        # TODO: post to DB backend
        # TODO: Refactor into configurable
        self.model_id = model_id 


    def round(self,config):
        """ """

        # TODO: Set / update reducer states and such

        # 1. Trigger Combiner nodes to compute an update, starting from the latest consensus model
        combiner_config = copy.deepcopy(config)
        combiner_config['rounds'] = 1
        combiner_config['task'] = 'training'

        for combiner in self.combiners:
            combiner.start(combiner_config)

        # 2. Reducer protocol - a single global model is formed from the combiner local models
        self.resolve()
    
        # 3. Trigger Combiner nodes to execute a validation round for the model
        combiner_config = copy.deepcopy(config)
        combiner_config['model_id'] = self.get_latest_model()
        combiner_config['task'] = 'validation'
        for combiner in self.combiners:
            combiner.start(combiner_config)


    def sync_combiners(self,model_id):
        """ 
            Spread the current consensus model to all active combiner nodes. 
            After execution all active combiners
            should be configured with identical model state. 
        """
        for combiner in self.combiners:
            response = combiner.set_model_id(model_id)

    def instruct(self, config):
        """ Main entrypoint, starts the control flow based on user-provided config (see Reducer class). """

        if self.__state == ReducerState.instructing:
            print("Already set in INSTRUCTING state", flush=True)
            return

        self.__state = ReducerState.instructing

        # TODO - move seeding from config to explicit step, use Reducer REST API reducer/seed/... ?
        if not self.get_latest_model():
            self.commit(config['model_id'])
            self.sync_combiners(self.get_latest_model())

        for round in range(config['rounds']): 
            self.round(config)
    
        self.__state = ReducerState.monitoring

    def reduce_random(self,model_ids):
        """ """
        import random
        model_id = random.sample(model_ids, 1)[0]
        return model_id

    def resolve(self):
        """ At the end of resolve, all combiners have the same model state. """

        # 1. Wait until all combiners report a local model that is ahead of the consensus global model

        # TODO: Use timeouts etc.
        ahead = []
        while len(ahead) < len(self.combiners):
          for combiner in self.combiners:
            model_id = combiner.get_latest_model()
            if model_id != self.get_latest_model():
                ahead.append(model_id)

        # 2. Aggregate models 
        # TODO: Do this with FedAvg strategy - we should delegate to the combiners to do this. 
        # For now we elect one combiner model update by random sampling and proceed with that one. 
        model_id = self.reduce_random(ahead)

        # 3. Propagate the new consensus model in the network
        self.spread_model(model_id)
        self.commit(model_id)

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
