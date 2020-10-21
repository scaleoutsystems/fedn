from fedn.clients.reducer.state import ReducerStateToString, StringToReducerState
from fedn.common.storage.db.mongo import connect_to_mongodb
from .reducerstatestore import ReducerStateStore


class MongoReducerStateStore(ReducerStateStore):
    def __init__(self, network_id, config, defaults=None):
        self.__inited = False
        try:
            self.config = config
            self.network_id = network_id
            self.mdb = connect_to_mongodb(self.config, self.network_id)
            self.state = self.mdb['state']
            self.models = self.mdb['models']
            self.latest_model = self.mdb['latest_model']
            self.compute_context = self.mdb['compute_context']
            self.compute_context_trail = self.mdb['compute_context_trail']
            self.network = self.mdb['network']
            self.reducer = self.network['reducer']
            self.combiners = self.network['combiners']
            self.clients = self.network['clients']
            self.storage = self.mdb['storage']
            self.status = self.mdb["status"]
            self.round_time = self.mdb["performances"]
            self.psutil_usage = self.mdb["psutil_usage"]

            self.__inited = True
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.state = None
            self.models = None
            self.latest_model = None
            self.compute_context = None
            self.compute_context_trail = None
            self.network = None
            self.combiners = None
            self.clients = None
            raise

        import yaml
        if defaults:
            with open(defaults, 'r') as file:
                try:
                    settings = dict(yaml.safe_load(file))
                    print(settings, flush=True)
                    self.transition(str(settings['state']))
                    if not self.get_latest():
                        self.set_latest(str(settings['model']))
                    else:
                        print("Model trail already exist - delete the entire trail if you want to reseed the system.",flush=True)
                    print("Setting filepath to {}".format(settings['context']), flush=True)
                    # self.set_compute_context(str())
                    # TODO Fix the ugly latering of indirection due to a bug in secure_filename returning an object with filename as attribute
                    # TODO fix with unboxing of value before storing and where consuming.
                    self.compute_context.update({'key': 'package'},
                                                {'$set': {'filename': {'filename': settings['context']}}}, True)
                    self.__inited = True
                except yaml.YamlError as e:
                    print(e)

    def is_inited(self):
        return self.__inited

    def get_config(self):
        data = {
            'type': 'MongoDB',
            'mongo_config': self.config,
            'network_id': self.network_id
        }
        return data

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
            if retcheck == '' or retcheck == ' ':  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def set_compute_context(self, filename):
        from datetime import datetime
        x = self.compute_context.update({'key': 'package'}, {'$set': {'filename': filename}}, True)
        self.compute_context_trail.update({'key': 'package'},
                                          {'$push': {'filename': filename, 'committed_at': str(datetime.now())}}, True)

    def get_compute_context(self):
        ret = self.compute_context.find({'key': 'package'})
        try:
            retcheck = ret[0]['filename']
            if retcheck == '' or retcheck == ' ':  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def get_model_info(self):
        # TODO: get all document in model collection...
        ret = self.models.find_one()
        try:
            if ret:
                committed_at = ret['committed_at']
                model = ret['model']
                model_dictionary = dict(zip(model, committed_at))
                return model_dictionary
            else:
                return None
        except (KeyError, IndexError):
            return None
    
    def get_storage_backend(self):
        """  """
        try:
            ret = self.storage.find({'status': 'enabled'}, projection={'_id': False})
            return ret[0]
        except (KeyError, IndexError):
            return None
    
    def set_storage_backend(self, config):
        """ """
        from datetime import datetime
        import copy
        config = copy.deepcopy(config)
        config['updated_at'] = str(datetime.now())
        config['status'] = 'enabled'
        ret = self.storage.update({'storage_type': config['storage_type']}, config, True)


    def set_reducer(self,reducer_data):
        """ """ 
        from datetime import datetime
        reducer_data['updated_at'] = str(datetime.now())
        ret = self.reducer.update({'name': reducer_data['name']}, reducer_data, True)

    def get_reducer(self):
        """ """
        try:
            ret = self.reducer.find_one()
            return ret
        except:
            return None

    def list_combiners(self):
        """ """ 
        combiners = []
        try:
            ret = self.combiner.find()
            return list(ret)
        except:
            return None

    def get_combiner(self,name):
        """ """
        try:
            ret = self.combiner.find({'key': name})
            return ret
        except:
            return None

    def set_combiner(self,combiner_data):s
        """ 
            Set or update combiner record. 
            combiner_data: dictionary, output of combiner.to_dict())
        """
        from datetime import datetime
        combiner_data['updated_at'] = str(datetime.now())
        ret = self.combiners.update({'name': combiner_data['name']}, combiner_data, True)
