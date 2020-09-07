from .state import ReducerState
import copy


import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
import grpc

from fedn.algo.fedavg import FEDAVGCombiner

class ReducerControl:

    def __init__(self):
        self.__state = ReducerState.idle
        self.combiners = []
        # TODO: Store in DB 
        self.model_id = None

    def get_model_id(self):
        # TODO: get from DB backend
        return self.model_id

    def set_model_id(self,model_id):
        # TODO: post to DB backend
        self.model_id = model_id 

    def round(self,config):
        """ """

        # 1. Spread the current global model to all combiners
        self.spread_model(self.get_model_id())

        # 2. Tell combiners to execute the compute plan / update the model
        combiner_config = copy.deepcopy(config)
        combiner_config['model_id'] = self.get_model_id()
        combiner_config['rounds'] = 1
        combiner_config['task'] = ''

        print("REDUCER: STARTING COMBINERS", flush=True)
        for combiner in self.combiners:
            print("REDUCER: STARTING {}".format(combiner.name), flush=True)
            combiner.start(combiner_config)
        print("REDUCER: STARTED {} COMBINERS".format(len(self.combiners), flush=True))

        # 3. Trigger reslution round - combiners aggregate their global models
        model_id = self.resolve()
        self.set_model_id(model_id)

        # 4. Trigger validation round 

    def spread_model(self,model_id):
        """ Spread the current consensus model_id to all combiner nodes. """
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

        for round in range(config['rounds']): 
            self.round(config)
    
        self.__state = ReducerState.monitoring

    def resolve(self):
        """ At the end of resolve, all combiners have the same model state. """

        ahead = []
        while len(ahead) < len(self.combiners):
          for combiner in self.combiners:
            model_id = combiner.get_model_id()
            if model_id != self.get_model_id():
                ahead.append(model_id)

        # TODO: Aggregate properly - we should find a way to delegate to the combiners to do this. 
        import random
        model_id = random.sample(ahead, 1)
        return model_id[0] 

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
