from .reducerstatestore import ReducerStateStore
from fedn.clients.reducer.state import (ReducerStateToString,
                                        StringToReducerState)
import redis

import pickle
r = redis.Redis(host='redis', port=6379, db=0)
def get_config(namespace,name, default_value=None):
    #if isinstance(value, dict):
    #    value = r.hgetall("fedn:config:{}:{}".format(namespace, name))
    #else:
    #    value = r.get("fedn:config:{}:{}".format(namespace, name))
    value = None
    try:
        temp = r.get("fedn:config:{}:{}".format(namespace, name))
        value = pickle.loads(temp)
        if value is None:
            return default_value
    except Exception as e:
        print("error no value for {}:{}".format(namespace,name), e, flush=True)
        return None
    return value

def set_config(namespace,name,value):
    status = None

    print("setting config for {}:{} and value {}".format(namespace, name, value), flush=True)
    if value is None:
        print("value is none ,why set to none?? for {}:{}".format(namespace, name), flush=True)
        return None
    try:
        p_value = pickle.dumps(value)
        status = r.set("fedn:config:{}:{}".format(namespace, name), p_value)
        #if isinstance(value, dict):
        #    status = r.hmset("fedn:config:{}:{}".format(namespace, name), value)
        #else:
        #    status = r.set("fedn:config:{}:{}".format(namespace, name), value)
    except Exception as e:
        print("Error:", e, flush=True)

    return status

def push_log(namespace, name, value):
    p_val = pickle.dumps(value)
    r.rpush("fedn:log:{}:{}".format(namespace, name), p_val)

def set_log(namespace, name, values):
    for i in range(0, r.llen("fedn:config:{}:{}".format(namespace, name))):
        r.lpop("fedn:config:{}:{}".format(namespace, name))
    for v in values:
        p_v = pickle.dumps(v)
        r.rpush("fedn:log:{}:{}".format(namespace, name), p_v)

def get_log(namespace, name):
    vals = r.lrange("fedn:log:{}:{}".format(namespace, name), 0, -1)
    return [pickle.loads(v) for v in vals]

def get_log_len(namespace, name):
    return r.llen("fedn:log:{}:{}".format(namespace, name))
class RedisReducerStateStore(ReducerStateStore):
    def __init__(self, network_id, config, defaults=None):
        self.__inited = False
        self.network_id = network_id
        #TODO move in the redis instantiation to a class varialbe and make the config from config dict.


        if not self.__inited:
            self.reducer = get_config('network','reducer')
            self.combiners = get_config('network','combiners')
            self.clients = get_config('network','clients')
            self.storage = get_config('network','storage')
            self.certificates = get_config('network','certificates')

            self.control_config = get_config('control','config')
            self.state = get_config('control','state')
            self.model = get_config('control','model')
            self.round = get_config('control','round')

            #self.status = get_config('control','status')
            self.round_time = get_config('control','round_time')
            self.psutil_monitoring = get_config('control','psutil_monitoring')
            self.combiner_round_time = get_config('control','combiner_round_time')

        self.__inited = True

    def is_inited(self):
        """

        :return:
        """
        return self.__inited

    #def get_config(self):

    #    data = {
    #       'type': 'Redis',
    #        'redis_config': self.config,
    #        'network_id': self.network_id
    #    }
    #    return data

    def state(self):
        """

        :return:
        """
        state = get_config('control', 'state')
        return StringToReducerState(state)

    def transition(self, state):
        """

        :param state:
        :return:
        """
        old_state = get_config('control', 'state')
        #old_state = self.state.find_one({'state': 'current_state'})
        if old_state != state:
            set_config('control', 'state', ReducerStateToString(state))
            #return self.state.update_one({'state': 'current_state'}, {'$set': {'state': ReducerStateToString(state)}},
            #                             True)
        else:
            print("Not updating state, already in {}".format(
                ReducerStateToString(state)))

    def set_latest(self, model_id):
        """

        :param model_id:
        """
        set_config('control', 'model', model_id)

        #self.model.update_one({'key': 'current_model'}, {
        #    '$set': {'model': model_id}}, True)
        from datetime import datetime
        data = { 'model': model_id, 'committed_at': str(datetime.now()) }
        #set_config('control', 'model', data)

        models = get_config('log', 'model_trail')
        if models is not None:
            models.append(data)
        else:
            models = [data]
        set_config('log', 'model_trail', models)
        #push_log('log','model_trail', data)
        #self.model.update_one({'key': 'model_trail'},
        #                      {'$push': {'model': model_id, 'committed_at': str(datetime.now())}},
        #                      True)

    def get_first(self):
        """ Return model_id for the latest model in the model_trail """
        models = get_config('log','model_trail')
        if models is not None:
            if len(models) > 0:

                return models[0]

        return None
        #ret = self.model.find_one({'key': 'model_trail'}, sort=[
        #    ("committed_at", pymongo.ASCENDING)])
        #if ret is None:
        #    return None

        #try:
        #    model_id = ret['model']
        #    if model_id == '' or model_id == ' ':  # ugly check for empty string
        #        return None
        #    return model_id
        #except (KeyError, IndexError):
        #    return None

    def get_latest(self):
        """ Return model_id for the latest model in the model_trail """
        return get_config('control', 'model', None)
        #models = get_config('log','model_trail')
        #if models is not None:
        #    if len(models) > 0:
        #        return models[len(models)-1]

        #ret = self.model.find_one({'key': 'current_model'})
        #if ret is None:
        #    return None

        #try:
        #    model_id = ret['model']
        #    if model_id == '' or model_id == ' ':  # ugly check for empty string
        #        return None
        #    return model_id
        #except (KeyError, IndexError):
        #    return None
        #return None

    def set_round_config(self, config):
        """

        :param config:
        """
        set_config('control', 'round_config', config)
        #self.control.config.update_one(
        #    {'key': 'round_config'}, {'$set': config}, True)

    def get_round_config(self):
        """

        :return:
        """
        return get_config('control', 'round_config')
        #ret = self.control.config.find({'key': 'round_config'})
        #try:
        #    retcheck = ret[0]
        #    if retcheck is None or retcheck == '' or retcheck == ' ':  # ugly check for empty string
        #        return None
        #    return retcheck
        #except (KeyError, IndexError):
        #    return None

    def set_compute_context(self, filename):
        """

        :param filename:
        """
        from datetime import datetime

        data =  {
            'filename': filename,
            'committed_at': str(datetime.now())
        }
        set_config('control', 'package', data)
        #set_config('log','package_trail', data)
        #self.control.config.update_one(
        #    {'key': 'package'}, {'$set': {'filename': filename}}, True)
        #self.control.config.update_one({'key': 'package_trail'},
        #                               {'$push': {'filename': filename, 'committed_at': str(datetime.now())}}, True)

    def get_compute_context(self):
        """

        :return:
        """
        return get_config('control', 'package')
        #ret = self.control.config.find({'key': 'package'})
        #try:
        #    retcheck = ret[0]
        #    if retcheck is None or retcheck == '' or retcheck == ' ':  # ugly check for empty string
        #        return None
        #    return retcheck
        #except (KeyError, IndexError):
        #    return None

    def set_framework(self, helper):
        """

        :param helper:
        """
        set_config('control', 'helper', helper)
        #self.control.config.update_one({'key': 'package'},
        #                               {'$set': {'helper': helper}}, True)

    def get_framework(self):
        """

        :return:
        """
        return get_config('control', 'helper')
        #ret = self.control.config.find_one({'key': 'package'})
        # if local compute package used, then 'package' is None
        #if not ret:
        #    # get framework from round_config instead
        #    ret = self.control.config.find_one({'key': 'round_config'})
        #print('FRAMEWORK:', ret)
        #try:
        #    retcheck = ret['helper']
        #    if retcheck == '' or retcheck == ' ':  # ugly check for empty string
        #        return None
        #    return retcheck
        #except (KeyError, IndexError):
        #    return None

    def get_model_info(self):
        """

        :return:
        """
        models = get_config('log','model_trail')
        return models[len(models)-1]

        #ret = self.model.find_one({'key': 'model_trail'})
        #try:
        #    if ret:
        #        committed_at = ret['committed_at']
        #        model = ret['model']
        #        model_dictionary = dict(zip(model, committed_at))
        #        return model_dictionary
        #    else:
        #        return None
        #except (KeyError, IndexError):
        #    return None

    #def get_events(self):
    #    """

    #   :return:
    #    """
    #    ret = self.control.status.find({})
    #    return ret

    def get_storage_backend(self):
        """  """
        return get_config('control', 'storage_backend')
        #try:
        #    ret = self.storage.find(
        #        {'status': 'enabled'}, projection={'_id': False})
        #    return ret[0]
        #except (KeyError, IndexError):
        #    return None

    def set_storage_backend(self, config):
        """ """
        return set_config('control', 'storage_backend', config)
        #config = copy.deepcopy(config)
        #config['updated_at'] = str(datetime.now())
        #config['status'] = 'enabled'
        #self.storage.update_one(
        #    {'storage_type': config['storage_type']}, {'$set': config}, True)

    def set_reducer(self, reducer_data):
        """ """
        from datetime import datetime
        print("set reduce 1", flush=True)
        reducer_data['updated_at'] = str(datetime.now())
        print("set reduce 2", flush=True)
        print("CHECK1: setting config for reducer {}".format(reducer_data), flush=True)
        set_config('control', 'reducer', reducer_data)

        config = get_config('control', 'reducer')
        print("CHECK2: got config for reducer {}".format(config), flush=True)

        print("set reduce 3", flush=True)
        #self.reducer.update_one({'name': reducer_data['name']}, {
        #    '$set': reducer_data}, True)

    def get_reducer(self):
        """ """
        return get_config('control', 'reducer')
        #try:
        #    ret = self.reducer.find_one()
        #    return ret
        #except Exception:
        #    return None

    #def list_combiners(self):
    #    """ """
    #    return get_log('control', 'combiners')
        #try:
        #    ret = self.combiners.find()
        #    return list(ret)
        #except Exception:
        #    return None

    def get_combiner(self, name):
        """ """
        combiners = get_config('control', 'combiners')
        if combiners is None or len(combiners) == 0:
            print("no combiners found", flush=True)
            return None
        for c in combiners:
            if c['name'] == name:
                print("found combiner with name {}".format(name), flush=True)
                return c

        print("no combiner found with name:{}".format(name), flush=True)
        return None
        #try:
        #    ret = self.combiners.find_one({'name': name})
        #    return ret
        #except Exception:
        #    return None

    def get_combiners(self):
        val = get_config('control', 'combiners')
        if val is None:
            return []
        return val

        #try:
        #    ret = self.combiners.find()
        #    return list(ret)
        #except Exception:
        #    return None

    def set_combiner(self, combiner_data):
        """
            Set or update combiner record.
            combiner_data: dictionary, output of combiner.to_dict())
        """
        combiners = get_config('control', 'combiners')
        from datetime import datetime
        combiner_data['updated_at'] = str(datetime.now())
        found = False
        if combiners is None or len(combiners) == 0:
            print("adding the first  combiner! {}".format(combiner_data), flush=True)
            combiners = []
            combiners.append(combiner_data)
        else:
            for c in combiners:
                if c['name'] == combiner_data['name']:
                    c.update(combiner_data)
                    found = True

        if not found:
            combiners.append(combiner_data)
        #combiners.update(combiner_data)
        set_config('control', 'combiners', combiners)

        #set_config('control', 'combiners', combiner_data)
        #self.combiners.update_one({'name': combiner_data['name']}, {
        #    '$set': combiner_data}, True)

    def delete_combiner(self, combiner):
        """ """
        new_combiners = []
        combiners = get_config('control', 'combiners')
        for c in combiners:
            if c['name'] != combiner:
                new_combiners.append(c)
        set_config('control', 'combiners', new_combiners)
        #try:
        #    self.combiners.delete_one({'name': combiner})
        #except Exception:
        #    print("WARNING, failed to delete combiner: {}".format(
        #        combiner), flush=True)

    def set_client(self, client_data):
        """
            Set or update client record.
            client_data: dictionarys
        """
        #client_data['updated_at'] = str(datetime.now())
        #self.clients.update_one({'name': client_data['name']}, {
        #    '$set': client_data}, True)
        clients = get_config('control', 'clients')
        from datetime import datetime
        client_data['updated_at'] = str(datetime.now())

        found = False
        if clients is None or len(clients) == 0:
            clients = []


        for c in clients:
            if c['name'] == client_data['name']:
                c.update(client_data)
                found = True

        if not found:
            clients.append(client_data)
        #combiners.update(combiner_data)
        set_config('control', 'clients', clients)
    def get_client(self, name):
        """ """
        clients = get_config('control', 'clients')
        if clients:
            for c in clients:
                if c['name'] == name:
                    return c
        return None
        #try:
        #    ret = self.clients.find({'key': name})
        #    if list(ret) == []:
        #        return None
        #    else:
        #        return ret
        #except Exception:
        #    return None

    def list_clients(self):
        """ """
        clients = get_config('control', 'clients')
        if clients is None:
            return []
        return clients
        #try:
        #     ret = self.clients.find()
        #     return list(ret)
        # except Exception:
        #     return None

    def drop_control(self):
        """ """
        # Control
        set_config('control', 'status', 'disabled')
        set_config('control', 'storage_backend', None)
        set_config('control', 'reducer', None)
        set_config('control', 'combiners', [])
        set_config('control', 'clients', [])
        #set_config('control', 'round_time', None)


        #self.state.drop()
        #self.control_config.drop()
        #self.control.drop()

        self.drop_models()

    def drop_models(self):
        """ """
        set_config('control', 'round', None)
        set_config('control', 'psutil_monitoring', None)
        set_config('control', 'model', None)
        set_config('control', 'combiner_round_time', None)
        #self.model.drop()
        #self.combiner_round_time.drop()
        #self.status.drop()
        #self.psutil_monitoring.drop()
        #self.round_time.drop()
        #self.round.drop()

    def update_client_status(self, client_data, status, role):
        """
            Set or update client status.
            assign roles to the active clients (trainer, validator, trainer-validator)
        """
        #self.clients.update_one({"name": client_data['name']},
        #                        {"$set":
        #                            {
        #                                "status": status,
        #                                "role": role
        #                            }
        #                        })
        #
        clients = get_config('control', 'clients')
        from datetime import datetime
        #client_data['updated_at'] = str(datetime.now())

        for c in clients:
            if c['name'] == client_data['name']:
                c.update({"status": status, "role": role})

        #combiners.update(combiner_data)
        set_config('control', 'clients', clients)
