import copy
from datetime import datetime

import pymongo
import yaml

from fedn.common.storage.db.mongo import connect_to_mongodb
from fedn.network.state import ReducerStateToString, StringToReducerState

from .statestorebase import StateStoreBase


class MongoStateStore(StateStoreBase):
    """

    """

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

            # Control
            self.control = self.mdb['control']
            self.package = self.control['package']
            self.state = self.control['state']
            self.model = self.control['model']
            self.sessions = self.control['sessions']
            self.rounds = self.control['rounds']

            # Logging
            self.status = self.control["status"]

            self.__inited = True
        except Exception as e:
            print("FAILED TO CONNECT TO MONGODB, {}".format(e), flush=True)
            self.state = None
            self.model = None
            self.control = None
            self.network = None
            self.combiners = None
            self.clients = None
            raise

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
                                print(
                                    "Model trail already initialized - refusing to overwrite from config. Purge model trail if you want to reseed the system.",
                                    flush=True)

                        if "context" in control:
                            print("Setting filepath to {}".format(
                                control['context']), flush=True)
                            # TODO Fix the ugly latering of indirection due to a bug in secure_filename returning an object with filename as attribute
                            # TODO fix with unboxing of value before storing and where consuming.
                            self.control.config.update_one({'key': 'package'},
                                                           {'$set': {'filename': control['context']}}, True)
                        if "helper" in control:
                            # self.set_framework(control['helper'])
                            pass

                        round_config = {'timeout': 180, 'validate': True}
                        try:
                            round_config['timeout'] = control['timeout']
                        except Exception:
                            pass

                        try:
                            round_config['validate'] = control['validate']
                        except Exception:
                            pass

                    # Storage settings
                    self.set_storage_backend(settings['storage'])

                    self.__inited = True
                except yaml.YAMLError as e:
                    print(e)

    def is_inited(self):
        """ Check if the statestore is intialized.

        :return:
        """
        return self.__inited

    def get_config(self):
        """Retrive the statestore config. 

        :return:
        """
        data = {
            'type': 'MongoDB',
            'mongo_config': self.config,
            'network_id': self.network_id
        }
        return data

    def state(self):
        """

        :return:
        """
        return StringToReducerState(self.state.find_one()['current_state'])

    def transition(self, state):
        """

        :param state:
        :return:
        """
        old_state = self.state.find_one({'state': 'current_state'})
        if old_state != state:
            return self.state.update_one({'state': 'current_state'}, {'$set': {'state': ReducerStateToString(state)}}, True)
        else:
            print("Not updating state, already in {}".format(
                ReducerStateToString(state)))

    def set_latest(self, model_id):
        """

        :param model_id:
        """

        self.model.update_one({'key': 'current_model'}, {
            '$set': {'model': model_id}}, True)
        self.model.update_one({'key': 'model_trail'}, {'$push': {'model': model_id, 'committed_at': str(datetime.now())}},
                              True)

    def get_first(self):
        """ Return model_id for the latest model in the model_trail """

        ret = self.model.find_one({'key': 'model_trail'}, sort=[
                                  ("committed_at", pymongo.ASCENDING)])
        if ret is None:
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
        if ret is None:
            return None

        try:
            model_id = ret['model']
            if model_id == '' or model_id == ' ':  # ugly check for empty string
                return None
            return model_id
        except (KeyError, IndexError):
            return None

    def get_latest_round(self):
        """ Get the id of the most recent round. """

        return self.rounds.find_one(sort=[("_id", pymongo.DESCENDING)])

    def get_round(self, id):
        """ Get round with id 'id'. """

        return self.rounds.find_one({'key': str(id)})

    # def set_round_config(self, config):
    #    """
    #
    #    :param config:
    #    """
    #    self.control.config.update_one(
    #        {'key': 'round_config'}, {'$set': config}, True)

    # def get_round_config(self):
    #    """
    #
    #    :return:
    #    """
    #    ret = self.control.config.find({'key': 'round_config'})
    #    try:
    #        retcheck = ret[0]
    #        if retcheck is None or retcheck == '' or retcheck == ' ':  # ugly check for empty string
    #            return None
    #        return retcheck
    #    except (KeyError, IndexError):
    #        return None

    def set_compute_package(self, filename):
        """ Set the active compute package.

        :param filename:
        """
        self.control.package.update_one(
            {'key': 'active'}, {'$set': {'filename': filename}}, True)
        self.control.package.update_one({'key': 'package_trail'},
                                        {'$push': {'filename': filename, 'committed_at': str(datetime.now())}}, True)

    def get_compute_package(self):
        """ Get the active compute package.

        :return:
        """
        ret = self.control.package.find({'key': 'active'})
        try:
            retcheck = ret[0]
            if retcheck is None or retcheck == '' or retcheck == ' ':  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def set_helper(self, helper):
        """

        :param helper:
        """
        self.control.package.update_one({'key': 'active'},
                                        {'$set': {'helper': helper}}, True)

    def get_helper(self):
        """

        :return:
        """
        ret = self.control.package.find_one({'key': 'active'})
        # if local compute package used, then 'package' is None
        # if not ret:
        # get framework from round_config instead
        #    ret = self.control.config.find_one({'key': 'round_config'})
        try:
            retcheck = ret['helper']
            if retcheck == '' or retcheck == ' ':  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def get_model_info(self):
        """

        :return:
        """
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
        """

        :return:
        """
        ret = self.control.status.find({})
        return ret

    def get_storage_backend(self):
        """  """
        try:
            ret = self.storage.find(
                {'status': 'enabled'}, projection={'_id': False})
            return ret[0]
        except (KeyError, IndexError):
            return None

    def set_storage_backend(self, config):
        """ """
        config = copy.deepcopy(config)
        config['updated_at'] = str(datetime.now())
        config['status'] = 'enabled'
        self.storage.update_one(
            {'storage_type': config['storage_type']}, {'$set': config}, True)

    def set_reducer(self, reducer_data):
        """ """
        reducer_data['updated_at'] = str(datetime.now())
        self.reducer.update_one({'name': reducer_data['name']}, {
            '$set': reducer_data}, True)

    def get_reducer(self):
        """ """
        try:
            ret = self.reducer.find_one()
            return ret
        except Exception:
            return None

    def list_combiners(self):
        """ """
        try:
            ret = self.combiners.find()
            return list(ret)
        except Exception:
            return None

    def get_combiner(self, name):
        """ """
        try:
            ret = self.combiners.find_one({'name': name})
            return ret
        except Exception:
            return None

    def get_combiners(self):
        """ """
        try:
            ret = self.combiners.find()
            return list(ret)
        except Exception:
            return None

    def set_combiner(self, combiner_data):
        """
            Set or update combiner record.
            combiner_data: dictionary, output of combiner.to_dict())
        """

        combiner_data['updated_at'] = str(datetime.now())
        self.combiners.update_one({'name': combiner_data['name']}, {
            '$set': combiner_data}, True)

    def delete_combiner(self, combiner):
        """ Delete a combiner entry. """
        try:
            self.combiners.delete_one({'name': combiner})
        except Exception:
            print("WARNING, failed to delete combiner: {}".format(
                combiner), flush=True)

    def set_client(self, client_data):
        """
            Set or update client record.
            client_data: dictionarys
        """
        client_data['updated_at'] = str(datetime.now())
        self.clients.update_one({'name': client_data['name']}, {
            '$set': client_data}, True)

    def get_client(self, name):
        """ Retrive a client record by name. """
        try:
            ret = self.clients.find({'key': name})
            if list(ret) == []:
                return None
            else:
                return ret
        except Exception:
            return None

    def list_clients(self):
        """List all clients registered on the network. """
        try:
            ret = self.clients.find()
            return list(ret)
        except Exception:
            return None

    def update_client_status(self, client_data, status, role):
        """
            Set or update client status.
            assign roles to the active clients (trainer, validator, trainer-validator)
        """
        self.clients.update_one({"name": client_data['name']},
                                {"$set":
                                    {
                                        "status": status,
                                        "role": role
                                    }
                                 })
