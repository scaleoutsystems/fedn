import copy
from datetime import datetime

import pymongo

from fedn.common.storage.db.mongo import connect_to_mongodb
from fedn.network.state import ReducerStateToString, StringToReducerState

from .statestorebase import StateStoreBase


class MongoStateStore(StateStoreBase):
    """ Statestore implementation using MongoDB.

    :param network_id: The network id.
    :type network_id: str
    :param config: The statestore configuration.
    :type config: dict
    :param defaults: The default configuration. Given by config/settings-reducer.yaml.template
    :type defaults: dict
    """

    def __init__(self, network_id, config, model_storage_config):
        """ Constructor."""
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

        # Storage settings
        self.set_storage_backend(model_storage_config)
        self.__inited = True       

    def is_inited(self):
        """ Check if the statestore is intialized.

        :return: True if initialized, else False.
        :rtype: bool
        """
        return self.__inited

    def get_config(self):
        """Retrive the statestore config.

        :return: The statestore config.
        :rtype: dict
        """
        data = {
            'type': 'MongoDB',
            'mongo_config': self.config,
            'network_id': self.network_id
        }
        return data

    def state(self):
        """ Get the current state.

        :return: The current state.
        :rtype: str
        """
        return StringToReducerState(self.state.find_one()['current_state'])

    def transition(self, state):
        """ Transition to a new state.

        :param state: The new state.
        :type state: str
        :return:
        """
        old_state = self.state.find_one({'state': 'current_state'})
        if old_state != state:
            return self.state.update_one({'state': 'current_state'}, {'$set': {'state': ReducerStateToString(state)}}, True)
        else:
            print("Not updating state, already in {}".format(
                ReducerStateToString(state)))

    def get_sessions(self):
        """ Get all sessions.

        :return: All sessions.
        :rtype: ObjectID
        """
        return self.sessions.find()

    def get_session(self, session_id):
        """ Get session with id.

        :param session_id: The session id.
        :type session_id: str
        :return: The session.
        :rtype: ObjectID
        """
        return self.sessions.find_one({'session_id': session_id})

    def set_latest_model(self, model_id):
        """ Set the latest model id.

        :param model_id: The model id.
        :type model_id: str
        :return:
        """

        self.model.update_one({'key': 'current_model'}, {
            '$set': {'model': model_id}}, True)
        self.model.update_one({'key': 'model_trail'}, {'$push': {'model': model_id, 'committed_at': str(datetime.now())}},
                              True)

    def get_initial_model(self):
        """ Return model_id for the initial model in the model trail

        :return: The initial model id. None if no model is found.
        :rtype: str
        """

        result = self.model.find_one({'key': 'model_trail'}, sort=[
            ("committed_at", pymongo.ASCENDING)])
        if result is None:
            return None

        try:
            model_id = result['model']
            if model_id == '' or model_id == ' ':
                return None
            return model_id[0]
        except (KeyError, IndexError):
            return None

    def get_latest_model(self):
        """ Return model_id for the latest model in the model_trail

        :return: The latest model id. None if no model is found.
        :rtype: str
        """
        result = self.model.find_one({'key': 'current_model'})
        if result is None:
            return None

        try:
            model_id = result['model']
            if model_id == '' or model_id == ' ':
                return None
            return model_id
        except (KeyError, IndexError):
            return None

    def get_latest_round(self):
        """ Get the id of the most recent round.

        :return: The id of the most recent round.
        :rtype: ObjectId
        """

        return self.rounds.find_one(sort=[("_id", pymongo.DESCENDING)])

    def get_round(self, id):
        """ Get round with id.

        :param id: id of round to get
        :type id: int
        :return: round with id, reducer and combiners
        :rtype: ObjectId
        """

        return self.rounds.find_one({'round_id': str(id)})

    def get_rounds(self):
        """ Get all rounds.

        :return: All rounds.
        :rtype: ObjectId
        """

        return self.rounds.find()

    def get_validations(self, **kwargs):
        """ Get validations from the database.

        :param kwargs: query to filter validations
        :type kwargs: dict
        :return: validations matching query
        :rtype: ObjectId
        """

        result = self.control.validations.find(kwargs)
        return result

    def set_compute_package(self, filename):
        """ Set the active compute package in statestore.

        :param filename: The filename of the compute package.
        :type filename: str
        :return: True if successful.
        :rtype: bool
        """
        self.control.package.update_one(
            {'key': 'active'}, {'$set': {'filename': filename, 'committed_at': str(datetime.now())}}, True)
        self.control.package.update_one({'key': 'package_trail'},
                                        {'$push': {'filename': filename, 'committed_at': str(datetime.now())}}, True)
        return True

    def get_compute_package(self):
        """ Get the active compute package.

        :return: The active compute package.
        :rtype: ObjectID
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
        """ Set the active helper package in statestore.

        :param helper: The name of the helper package. See helper.py for available helpers.
        :type helper: str
        :return:
        """
        self.control.package.update_one({'key': 'active'},
                                        {'$set': {'helper': helper}}, True)

    def get_helper(self):
        """ Get the active helper package.

        :return: The active helper set for the package.
        :rtype: str
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

    def get_model_trail(self):
        """ Get the model trail.

        :return: dictionary of model_id: committed_at
        :rtype: dict
        """
        result = self.model.find_one({'key': 'model_trail'})
        try:
            if result is not None:
                committed_at = result['committed_at']
                model = result['model']
                model_dictionary = dict(zip(model, committed_at))
                return model_dictionary
            else:
                return None
        except (KeyError, IndexError):
            return None

    def get_events(self, **kwargs):
        """ Get events from the database.

        :param kwargs: query to filter events
        :type kwargs: dict
        :return: events matching query
        :rtype: ObjectId
        """
        # check if kwargs is empty
        if not kwargs:
            return self.control.status.find()
        else:
            result = self.control.status.find(kwargs)
        return result

    def get_storage_backend(self):
        """ Get the storage backend.

        :return: The storage backend.
        :rtype: ObjectID
        """
        try:
            ret = self.storage.find(
                {'status': 'enabled'}, projection={'_id': False})
            return ret[0]
        except (KeyError, IndexError):
            return None

    def set_storage_backend(self, config):
        """ Set the storage backend.

        :param config: The storage backend configuration.
        :type config: dict
        :return:
        """
        config = copy.deepcopy(config)
        config['updated_at'] = str(datetime.now())
        config['status'] = 'enabled'
        self.storage.update_one(
            {'storage_type': config['storage_type']}, {'$set': config}, True)

    def set_reducer(self, reducer_data):
        """ Set the reducer in the statestore.

        :param reducer_data: dictionary of reducer config.
        :type reducer_data: dict
        :return:
        """
        reducer_data['updated_at'] = str(datetime.now())
        self.reducer.update_one({'name': reducer_data['name']}, {
            '$set': reducer_data}, True)

    def get_reducer(self):
        """ Get reducer.config.

        return: reducer config.
        rtype: ObjectId
        """
        try:
            ret = self.reducer.find_one()
            return ret
        except Exception:
            return None

    def get_combiner(self, name):
        """ Get combiner by name.

        :return: The combiner.
        :rtype: ObjectId
        """
        try:
            ret = self.combiners.find_one({'name': name})
            return ret
        except Exception:
            return None

    def get_combiners(self):
        """ Get all combiners.

        :return: list of combiners.
        :rtype: list
        """
        try:
            ret = self.combiners.find()
            return list(ret)
        except Exception:
            return None

    def set_combiner(self, combiner_data):
        """ Set combiner in statestore.

        :param combiner_data: dictionary of combiner config
        :type combiner_data: dict
        :return:
        """

        combiner_data['updated_at'] = str(datetime.now())
        self.combiners.update_one({'name': combiner_data['name']}, {
            '$set': combiner_data}, True)

    def delete_combiner(self, combiner):
        """ Delete a combiner from statestore.

        :param combiner: name of combiner to delete.
        :type combiner: str
        :return:
        """
        try:
            self.combiners.delete_one({'name': combiner})
        except Exception:
            print("WARNING, failed to delete combiner: {}".format(
                combiner), flush=True)

    def set_client(self, client_data):
        """ Set client in statestore.

        :param client_data: dictionary of client config.
        :type client_data: dict
        :return:
        """
        client_data['updated_at'] = str(datetime.now())
        self.clients.update_one({'name': client_data['name']}, {
            '$set': client_data}, True)

    def get_client(self, name):
        """ Get client by name.

        :param name: name of client to get.
        :type name: str
        :return: The client. None if not found.
        :rtype: ObjectId
        """
        try:
            ret = self.clients.find({'key': name})
            if list(ret) == []:
                return None
            else:
                return ret
        except Exception:
            return None

    def list_clients(self):
        """List all clients registered on the network.

        :return: list of clients.
        :rtype: list(ObjectId)
        """
        try:
            ret = self.clients.find()
            return list(ret)
        except Exception:
            return None

    def update_client_status(self, client_data, status, role):
        """ Set or update client status.

        :param client_data: dictionary of client config.
        :type client_data: dict
        :param status: status of client.
        :type status: str
        :param role: role of client.
        :type role: str
        :return:
        """
        self.clients.update_one({"name": client_data['name']},
                                {"$set":
                                    {
                                        "status": status,
                                        "role": role
                                    }
                                 })
