import copy
import uuid
from datetime import datetime

import pymongo
from google.protobuf.json_format import MessageToDict

from fedn.common.log_config import logger
from fedn.network.state import ReducerStateToString, StringToReducerState


class MongoStateStore:
    """Statestore implementation using MongoDB.

    :param network_id: The network id.
    :type network_id: str
    :param config: The statestore configuration.
    :type config: dict
    :param defaults: The default configuration. Given by config/settings-reducer.yaml.template
    :type defaults: dict
    """

    def __init__(self, network_id, config):
        """Constructor."""
        self.__inited = False
        try:
            self.config = config
            self.network_id = network_id
            self.mdb = self.connect()

            # FEDn network
            self.network = self.mdb["network"]
            self.reducer = self.network["reducer"]
            self.combiners = self.network["combiners"]
            self.clients = self.network["clients"]
            self.storage = self.network["storage"]

            # Control
            self.control = self.mdb["control"]
            self.package = self.control["package"]
            self.state = self.control["state"]
            self.model = self.control["model"]
            self.sessions = self.control["sessions"]
            self.rounds = self.control["rounds"]
            self.validations = self.control["validations"]

            # Logging
            self.status = self.control["status"]

            self.__inited = True
        except Exception as e:
            logger.error("FAILED TO CONNECT TO MONGODB, {}".format(e))
            self.state = None
            self.model = None
            self.control = None
            self.network = None
            self.combiners = None
            self.clients = None
            raise

        self.init_index()

    def connect(self):
        """Establish client connection to MongoDB.

        :param config: Dictionary containing connection strings and security credentials.
        :type config: dict
        :param network_id: Unique identifier for the FEDn network, used as db name
        :type network_id: str
        :return: MongoDB client pointing to the db corresponding to network_id
        """
        try:
            mc = pymongo.MongoClient(**self.config)
            # This is so that we check that the connection is live
            mc.server_info()
            mdb = mc[self.network_id]
            return mdb
        except Exception:
            raise

    def init_index(self):
        self.package.create_index([("id", pymongo.DESCENDING)])
        self.clients.create_index([("client_id", pymongo.DESCENDING)])

    def is_inited(self):
        """Check if the statestore is intialized.

        :return: True if initialized, else False.
        :rtype: bool
        """
        return self.__inited

    def state(self):
        """Get the current state.

        :return: The current state.
        :rtype: str
        """
        return StringToReducerState(self.state.find_one()["current_state"])

    def get_compute_package(self):
        """Get the active compute package.

        :return: The active compute package.
        :rtype: ObjectID
        """
        try:
            find = {"key": "active"}
            projection = {"key": False, "_id": False}
            ret = self.control.package.find_one(find, projection)
            return ret
        except Exception as e:
            logger.error("ERROR: {}".format(e))
            return None

    def get_model(self, model_id):
        """Get model with id.

        :param model_id: id of model to get
        :type model_id: str
        :return: model with id
        :rtype: ObjectId
        """
        return self.model.find_one({"key": "models", "model": model_id})

    def get_storage_backend(self):
        """Get the storage backend.

        :return: The storage backend.
        :rtype: ObjectID
        """
        try:
            ret = self.storage.find({"status": "enabled"}, projection={"_id": False})
            return ret[0]
        except (KeyError, IndexError):
            return None

    def set_storage_backend(self, config):
        """Set the storage backend.

        :param config: The storage backend configuration.
        :type config: dict
        :return:
        """
        config = copy.deepcopy(config)
        config["updated_at"] = str(datetime.now())
        config["status"] = "enabled"
        self.storage.update_one({"storage_type": config["storage_type"]}, {"$set": config}, True)
