from .state import ReducerState
import copy


class ReducerControl:

    def __init__(self):
        self.__state = ReducerState.idle
        self.combiners = []
        # TODO: Store in DB 
        self.model_id = None

    def get_model_id(self):
        # TODO: get single point of thruth from DB / Eth backend
        return self.model_id

    def set_model_id(self,model_id):
        # TODO: post to DB backend
        self.model_id = model_id 

    def round(self,config):
        """ """

        # TODO: Set / update reducer states and such

        # 1. Tell combiners to execute the compute plan / update the combiner model
        combiner_config = copy.deepcopy(config)
        combiner_config['rounds'] = 1
        combiner_config['task'] = ''

        print("REDUCER: STARTING COMBINERS", flush=True)
        for combiner in self.combiners:
            print("REDUCER: STARTING {}".format(combiner.name), flush=True)
            combiner.start(combiner_config)
        print("REDUCER: STARTED {} COMBINERS".format(len(self.combiners), flush=True))

        # 2. Reducer protocol - a single global model is formed from the combiner local models
        self.resolve()
    
        # 3. Trigger validation round 
        combiner_config = copy.deepcopy(config)
        combiner_config['task'] = 'validation'
        for combiner in self.combiners:
            combiner_config = copy.deepcopy(config)
            combiner_config['task'] = 'validation'
            combiner.start(combiner_config)


    def spread_model(self,model_id):
        """ 
            Spread the current consensus model to all combiner nodes. 
            After sucessful execution, all active combiners
            should be configured with identical model state. 
        """
        for combiner in self.combiners:
            response = combiner.set_model_id(model_id)
            print("REDUCER_CONTROL: Setting model_ids: {}".format(response),flush=True)

    def instruct(self, config):
        if self.__state == ReducerState.instructing:
            print("Already set in INSTRUCTING state", flush=True)
            return

        self.__state = ReducerState.instructing

        # TODO - move seeding from config to explicit step, use Reducer REST API reducer/seed/... ?
        if not self.get_model_id():
            self.set_model_id(config['model_id'])
            self.spread_model(self.get_model_id())

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
            model_id = combiner.get_model_id()
            if model_id != self.get_model_id():
                ahead.append(model_id)

        # 2. Aggregate models 
        # TODO: Do this with FedAvg strategy - we should delegate to the combiners to do this. 
        # For now we elect one combiner local model by random sampling and proceed with that one. 
        model_id = self.reduce_random(ahead)

        # 3. Propagate the new consensus model in the network
        self.spread_model(model_id)
        self.set_model_id(model_id)

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
