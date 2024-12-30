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

    def set_latest_model(self, model_id, session_id=None):
        """Set the latest model id.

        :param model_id: The model id.
        :type model_id: str
        :return:
        """
        committed_at = datetime.now()
        current_model = self.model.find_one({"key": "current_model"})
        parent_model = None

        # if session_id is set the it means the model is generated from a session
        # and we need to set the parent model
        # if not the model is uploaded by the user and we don't need to set the parent model
        if session_id is not None:
            parent_model = current_model["model"] if current_model and "model" in current_model else None

        self.model.insert_one(
            {
                "key": "models",
                "model": model_id,
                "parent_model": parent_model,
                "session_id": session_id,
                "committed_at": committed_at,
            }
        )

        self.model.update_one({"key": "current_model"}, {"$set": {"model": model_id}}, True)
        self.model.update_one(
            {"key": "model_trail"},
            {
                "$push": {
                    "model": model_id,
                    "committed_at": str(committed_at),
                }
            },
            True,
        )

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

    def set_round_config(self, round_id, round_config):
        """Set round configuration.

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_config: The round configuration
        :type round_config: dict
        """
        self.rounds.update_one({"round_id": round_id}, {"$set": {"round_config": round_config}}, True)

    def set_round_status(self, round_id, round_status):
        """Set round status.

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_status: The status of the round.
        """
        self.rounds.update_one({"round_id": round_id}, {"$set": {"status": round_status}}, True)

    def set_round_data(self, round_id, round_data):
        """Update round metadata

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_data: The round metadata
        :type round_data: dict
        """
        self.rounds.update_one({"round_id": round_id}, {"$set": {"round_data": round_data}}, True)

    def update_client_status(self, clients, status):
        """Update client status in statestore.
        :param client_name: The client name
        :type client_name: str
        :param status: The client status
        :type status: str
        :return: None
        """
        datetime_now = datetime.now()
        filter_query = {"client_id": {"$in": clients}}

        update_query = {"$set": {"last_seen": datetime_now, "status": status}}
        self.clients.update_many(filter_query, update_query)
