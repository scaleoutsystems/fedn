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

            # FEDn network
            self.network = self.mdb['network']
            self.reducer = self.network['reducer']
            self.combiners = self.network['combiners']
            self.clients = self.network['clients']
            self.storage = self.network['storage']
            self.certificates = self.network['certificates']
            # Control 
            self.control = self.mdb['control']
            self.control_config = self.control['config']
            self.state = self.control['state']
            self.model = self.control['model']
            self.round = self.control["round"]

            # Logging and dashboards
            self.status = self.control["status"]
            self.round_time = self.control["round_time"]
            self.psutil_monitoring = self.control["psutil_monitoring"]
            self.combiner_round_time = self.control['combiner_round_time']



            self.__inited = True
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.state = None
            self.model = None
            self.control = None
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

                    # Control settings
                    if "control" in settings and settings["control"]:
                        control = settings['control']
                        try:
                            self.transition(str(control['state']))
                        except KeyError:
                            self.transition("idle")

                        if "model" in control:
                            if not self.get_latest():
                                self.set_latest(str(control['model']))
                            else:
                                print("Model trail already initialized - refusing to overwrite from config. Purge model trail if you want to reseed the system.",flush=True)
                    
                        if "context" in control:
                            print("Setting filepath to {}".format(control['context']), flush=True)
                            # TODO Fix the ugly latering of indirection due to a bug in secure_filename returning an object with filename as attribute
                            # TODO fix with unboxing of value before storing and where consuming.
                            self.control.config.update({'key': 'package'},
                                                        {'$set': {'filename': control['context']}}, True)
                        if "helper" in control:
                            #self.set_framework(control['helper'])
                            pass

                        round_config = {'timeout':180, 'validate':True}
                        try:
                            round_config['timeout'] = control['timeout']
                        except:
                            pass

                        try:
                            round_config['validate'] = control['validate']
                        except:
                            pass
  

                    # Storage settings
                    self.set_storage_backend(settings['storage'])


                    self.__inited = True
                except yaml.YAMLError as e:
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
        x = self.model.update({'key': 'current_model'}, {'$set': {'model': model_id}}, True)
        self.model.update({'key': 'model_trail'}, {'$push': {'model': model_id, 'committed_at': str(datetime.now())}}, True)

    def get_first(self):
        """ Return model_id for the latest model in the model_trail """
        import pymongo
        ret = self.model.find_one({'key': 'model_trail'}, sort=[("committed_at", pymongo.ASCENDING)])
        if ret == None:
            return None

        try:
            model_id = ret['model']
            if model_id == '' or model_id == ' ':  # ugly check for empty string
                return None
            return model_id
        except (KeyError, IndexError):
            return None

    def get_latest(self):
        """ Return model_id for the latest model in the model_trail """
        ret = self.model.find_one({'key': 'current_model'})
        if ret == None:
            return None

        try:
            model_id = ret['model']
            if model_id == '' or model_id == ' ':  # ugly check for empty string
                return None
            return model_id
        except (KeyError, IndexError):
            return None

    def set_round_config(self, config):
        from datetime import datetime
        x = self.control.config.update({'key': 'round_config'}, {'$set': config}, True)

    def get_round_config(self):
        ret = self.control.config.find({'key': 'round_config'})
        try:
            retcheck = ret[0]
            if retcheck == None or retcheck == '' or retcheck == ' ':  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def set_compute_context(self, filename):
        from datetime import datetime
        x = self.control.config.update({'key': 'package'}, {'$set': {'filename': filename}}, True)
        self.control.config.update({'key': 'package_trail'},
                                          {'$push': {'filename': filename, 'committed_at': str(datetime.now())}}, True)

    def get_compute_context(self):
        ret = self.control.config.find({'key': 'package'})
        try:
            retcheck = ret[0]
            if retcheck == None or retcheck == '' or retcheck == ' ':  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def set_framework(self, helper):
        self.control.config.update({'key': 'package'},
                                    {'$set': {'helper': helper}}, True)

    def get_framework(self):
        ret = self.control.config.find({'key': 'package'})
        try:
            retcheck = ret[0]['helper']
            if retcheck == '' or retcheck == ' ':  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def get_model_info(self):
        ret = self.model.find_one({'key': 'model_trail'})
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

    def get_events(self):
        ret = self.control.status.find({})
        return ret

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
        try:
            ret = self.combiners.find()
            return list(ret)
        except:
            return None

    def get_combiner(self,name):
        """ """
        try:
            ret = self.combiners.find_one({'name': name})
            return ret
        except:
            return None

    def get_combiners(self):
        """ """
        try:
            ret = self.combiners.find()
            return list(ret)
        except:
            return None      


    def set_combiner(self,combiner_data):
        """ 
            Set or update combiner record. 
            combiner_data: dictionary, output of combiner.to_dict())
        """
        from datetime import datetime
        combiner_data['updated_at'] = str(datetime.now())
        ret = self.combiners.update({'name': combiner_data['name']}, combiner_data, True)

    def delete_combiner(self,combiner):
        """ """
        try:
            self.combiners.delete_one({'name': combiner})
        except:
            print("WARNING, failed to delete combiner: {}".format(combiner), flush=True)

    def set_client(self, client_data):
        """ 
            Set or update client record. 
            client_data: dictionarys
        """
        from datetime import datetime
        client_data['updated_at'] = str(datetime.now())
        ret = self.clients.update({'name': client_data['name']}, client_data, True)

    def get_client(self, name):
        """ """
        try:
            ret = self.clients.find({'key': name})
            if list(ret) == []:
                return None
            else:
                return ret
        except:
            return None

    def list_clients(self):
        """ """ 
        try:
            ret = self.clients.find()
            return list(ret)
        except:
            return None

    def drop_control(self):
        """ """
        # Control 
        self.state.drop() 
        self.control_config.drop()
        self.control.drop()

        self.drop_models() 


    def drop_models(self):
        """ """
        self.model.drop()
        self.combiner_round_time.drop()
        self.status.drop()
        self.psutil_monitoring.drop()
        self.round_time.drop()
        self.round.drop()





    
