import copy
import os
import tempfile
import time

from fedn.utils.helpers import KerasSequentialHelper

from .state import ReducerState


class Model:
    """ (DB) representation of a global model. """

    def __init__(self, id=None, model_type="Keras"):
        self.id = id
        self.name = ""
        self.type = model_type
        self.version = ""
        self.parent = ""
        self.alliance_uid = ""
        self.round_id = 0


class ReducerControl:

    def __init__(self, statestore):
        self.__state = ReducerState.idle
        self.statestore = statestore
        #self.statestore.set_latest(None)
        self.combiners = []
        # TODO: Use DB / Eth
        # models should be an immutable, ordered chain of global models
        # self.strategy = ReducerControlStrategy(config["startegy"])

        # TODO remove temporary hardcoded config of storage persistance backend
        s3_config = {'storage_access_key': os.environ['FEDN_MINIO_ACCESS_KEY'],
                     'storage_secret_key': os.environ['FEDN_MINIO_SECRET_KEY'],
                     'storage_bucket': 'models',
                     'storage_secure_mode': False,
                     'storage_hostname': os.environ['FEDN_MINIO_HOST'],
                     'storage_port': int(os.environ['FEDN_MINIO_PORT'])}

        from fedn.common.storage.s3.s3repo import S3ModelRepository
        self.model_repository = S3ModelRepository(s3_config)
        self.bucket_name = s3_config["storage_bucket"]

    def get_latest_model(self):
        # TODO: get single point of thruth from DB / Eth backend
        return self.statestore.get_latest()


    def commit(self, model_id, model=None):
        """ Commit a model. This establishes this model as the lastest consensus model. """

        # TODO: post to DB backend
        # TODO: Refactor into configurable

        # TODO: This is because we start with a manually uploaded seed model, unify and move seeding of
        # model chain to own method.         
        if model:
            fod, outfile_name = tempfile.mkstemp(suffix='.h5')
            model.save(outfile_name)
            model_id = self.model_repository.set_model(outfile_name, is_file=True)
            os.unlink(outfile_name)

        # TODO: Append to model chain in DB backend
        self.statestore.set_latest(model_id)


    def _out_of_sync(self):
        osync = []
        for combiner in self.combiners:
            model_id = combiner.get_model_id()
            if model_id != self.get_latest_model():
                osync.append(combiner)
        return osync

    def round(self, config):
        """ """

        # TODO: Set / update reducer states and such
        if len(self.combiners) < 1:
            print("REDUCER: No combiners connected!")
            return

        # 1. Trigger Combiners to compute an update, starting from the latest consensus model.
        combiner_config = copy.deepcopy(config)
        combiner_config['rounds'] = 1
        combiner_config['task'] = 'training'
        combiner_config['model_id'] = self.get_latest_model()
        for combiner in self.combiners:
            combiner.start(combiner_config)

        # Wait until all combiners are out of sync with the current global model, or we timeout.
        wait = 0.0
        while len(self._out_of_sync()) < len(self.combiners):
            time.sleep(1.0)
            wait += 1.0
            if wait >= config['round_timeout']:
                break

        # 2. Resolver protocol - a single global model is formed from the combiner local models.
        self.resolve()

        # 3. Trigger Combiner nodes to execute a validation round for the current model
        combiner_config = copy.deepcopy(config)
        combiner_config['model_id'] = self.get_latest_model()
        combiner_config['task'] = 'validation'
        for combiner in self.combiners:
            combiner.start(combiner_config)

    def sync_combiners(self, model_id):

        if not model_id:
            print("GOT NO MODEL TO SET!", flush=True)
            return
        """ 
            Spread the current consensus model to all active combiner nodes. 
            After execution all active combiners
            should be configured with identical model state. 
        """

        # TODO: We should only be able to set the active model on the Combiner
        # if the combiner is in IDLE state. 
        for combiner in self.combiners:
            response = combiner.set_model_id(model_id)
            print("REDUCER_CONTROL: Setting model_ids: {}".format(response), flush=True)

    def instruct(self, config):
        """ Main entrypoint, executes the compute plan. """

        if self.__state == ReducerState.instructing:
            print("Already set in INSTRUCTING state", flush=True)
            return

        self.__state = ReducerState.instructing

        # TODO - move seeding from config to explicit step, use Reducer REST API reducer/seed/... ?
        if not self.get_latest_model():
            self.commit(config['model_id'])
            self.sync_combiners(self.get_latest_model())

        self.__state = ReducerState.monitoring
        for round in range(int(config['rounds'])):
            self.round(config)


        self.__state = ReducerState.idle

    def reduce(self, combiners):
        """ Combine current models at Combiner nodes into one global model. """
        import uuid
        model_id = uuid.uuid4()

        # TODO: Make configurable
        helper = KerasSequentialHelper()

        for i, combiner in enumerate(combiners,1):
            data = combiner.get_model()
            if i == 1:
                model = helper.load_model(data.getbuffer())
            else:
                model_next = helper.load_model(combiner.get_model().getbuffer())
                helper.increment_average(model, model_next, i)

        return model, model_id

    def reduce_random(self, combiners):
        """ This is only used for debugging purposes. s"""
        import random
        combiner = random.sample(combiners, 1)[0]
        import uuid
        model_id = uuid.uuid4()
        helper = KerasSequentialHelper()
        return helper.load_model(combiner.get_model().getbuffer()),model_id

    def resolve(self):
        """ At the end of resolve, all combiners have the same model state. """

        # 1. Aggregate models from all combiners that are out of sync
        combiners = self._out_of_sync()
        if len(combiners) > 0:
            model, model_id = self.reduce(combiners)

        # 2. Commit the new consensus model to the chain and propagate it in the Combiner network
        self.commit(model_id, model)
        self.sync_combiners(self.get_latest_model())



    def monitor(self, config=None):
        if self.__state == ReducerState.monitoring:
            print("monitoring")

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
