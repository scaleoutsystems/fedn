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

    def get_config(self):
        """Retrive the statestore config.

        :return: The statestore config.
        :rtype: dict
        """
        data = {
            "type": "MongoDB",
            "mongo_config": self.config,
            "network_id": self.network_id,
        }
        return data

    def state(self):
        """Get the current state.

        :return: The current state.
        :rtype: str
        """
        return StringToReducerState(self.state.find_one()["current_state"])

    def transition(self, state):
        """Transition to a new state.

        :param state: The new state.
        :type state: str
        :return:
        """
        old_state = self.state.find_one({"state": "current_state"})
        if old_state != state:
            return self.state.update_one(
                {"state": "current_state"},
                {"$set": {"state": ReducerStateToString(state)}},
                True,
            )
        else:
            logger.info("Not updating state, already in {}".format(ReducerStateToString(state)))

    def get_sessions(self, limit=None, skip=None, sort_key="_id", sort_order=pymongo.DESCENDING):
        """Get all sessions.

        :param limit: The maximum number of sessions to return.
        :type limit: int
        :param skip: The number of sessions to skip.
        :type skip: int
        :param sort_key: The key to sort by.
        :type sort_key: str
        :param sort_order: The sort order.
        :type sort_order: pymongo.ASCENDING or pymongo.DESCENDING
        :return: Dictionary of sessions in result (array of session objects) and count.
        """
        result = None

        if limit is not None and skip is not None:
            limit = int(limit)
            skip = int(skip)

            result = self.sessions.find().limit(limit).skip(skip).sort(sort_key, sort_order)
        else:
            result = self.sessions.find().sort(sort_key, sort_order)

        count = self.sessions.count_documents({})

        return {
            "result": result,
            "count": count,
        }

    def get_session(self, session_id):
        """Get session with id.

        :param session_id: The session id.
        :type session_id: str
        :return: The session.
        :rtype: ObjectID
        """
        return self.sessions.find_one({"session_id": session_id})

    def get_session_status(self, session_id):
        """Get the session status.

        :param session_id: The session id.
        :type session_id: str
        :return: The session status.
        :rtype: str
        """
        session = self.sessions.find_one({"session_id": session_id})
        return session["status"]

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

    def get_initial_model(self):
        """Return model_id for the initial model in the model trail

        :return: The initial model id. None if no model is found.
        :rtype: str
        """
        result = self.model.find_one({"key": "model_trail"}, sort=[("committed_at", pymongo.ASCENDING)])
        if result is None:
            return None

        try:
            model_id = result["model"]
            if model_id == "" or model_id == " ":
                return None
            return model_id[0]
        except (KeyError, IndexError):
            return None

    def get_latest_model(self):
        """Return model_id for the latest model in the model_trail

        :return: The latest model id. None if no model is found.
        :rtype: str
        """
        result = self.model.find_one({"key": "current_model"})
        if result is None:
            return None

        try:
            model_id = result["model"]
            if model_id == "" or model_id == " ":
                return None
            return model_id
        except (KeyError, IndexError):
            return None

    def set_current_model(self, model_id: str):
        """Set the current model in statestore.

        :param model_id: The model id.
        :type model_id: str
        :return:
        """
        try:
            committed_at = datetime.now()

            existing_model = self.model.find_one({"key": "models", "model": model_id})

            if existing_model is not None:
                self.model.update_one({"key": "current_model"}, {"$set": {"model": model_id, "committed_at": committed_at, "session_id": None}}, True)

                return True
        except Exception as e:
            logger.error("ERROR: {}".format(e))

        return False

    def get_latest_round(self):
        """Get the id of the most recent round.

        :return: The id of the most recent round.
        :rtype: ObjectId
        """
        return self.rounds.find_one(sort=[("_id", pymongo.DESCENDING)])

    def get_round(self, id):
        """Get round with id.

        :param id: id of round to get
        :type id: int
        :return: round with id, reducer and combiners
        :rtype: ObjectId
        """
        return self.rounds.find_one({"round_id": str(id)})

    def get_rounds(self):
        """Get all rounds.

        :return: All rounds.
        :rtype: ObjectId
        """
        return self.rounds.find()

    def get_validations(self, **kwargs):
        """Get validations from the database.

        :param kwargs: query to filter validations
        :type kwargs: dict
        :return: validations matching query
        :rtype: ObjectId
        """
        result = self.control.validations.find(kwargs)
        return result

    def set_active_compute_package(self, id: str):
        """Set the active compute package in statestore.

        :param id: The id of the compute package (not document _id).
        :type id: str
        :return: True if successful.
        :rtype: bool
        """
        try:
            find = {"id": id}
            projection = {"_id": False, "key": False}

            doc = self.control.package.find_one(find, projection)

            if doc is None:
                return False

            doc["key"] = "active"

            self.control.package.replace_one({"key": "active"}, doc)

        except Exception as e:
            logger.error("ERROR: {}".format(e))
            return False

        return True

    def set_compute_package(self, file_name: str, storage_file_name: str, helper_type: str, name: str = None, description: str = None):
        """Set the active compute package in statestore.

        :param file_name: The file_name of the compute package.
        :type file_name: str
        :return: True if successful.
        :rtype: bool
        """
        obj = {
            "file_name": file_name,
            "storage_file_name": storage_file_name,
            "helper": helper_type,
            "committed_at": datetime.now(),
            "name": name,
            "description": description,
            "id": str(uuid.uuid4()),
        }

        self.control.package.update_one(
            {"key": "active"},
            {"$set": obj},
            True,
        )

        trail_obj = {**{"key": "package_trail"}, **obj}

        self.control.package.insert_one(trail_obj)

        return True

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

    def list_compute_packages(self, limit: int = None, skip: int = None, sort_key="committed_at", sort_order=pymongo.DESCENDING):
        """List compute packages in the statestore (paginated).

        :param limit: The maximum number of compute packages to return.
        :type limit: int
        :param skip: The number of compute packages to skip.
        :type skip: int
        :param sort_key: The key to sort by.
        :type sort_key: str
        :param sort_order: The sort order.
        :type sort_order: pymongo.ASCENDING or pymongo.DESCENDING
        :return: Dictionary of compute packages in result and count.
        :rtype: dict
        """
        result = None
        count = None

        find_option = {"key": "package_trail"}
        projection = {"key": False, "_id": False}

        try:
            if limit is not None and skip is not None:
                result = self.control.package.find(find_option, projection).limit(limit).skip(skip).sort(sort_key, sort_order)
            else:
                result = self.control.package.find(find_option, projection).sort(sort_key, sort_order)

            count = self.control.package.count_documents(find_option)

        except Exception as e:
            logger.error("ERROR: {}".format(e))
            return None

        return {
            "result": result or [],
            "count": count or 0,
        }

    def set_helper(self, helper):
        """Set the active helper package in statestore.

        :param helper: The name of the helper package. See helper.py for available helpers.
        :type helper: str
        :return:
        """
        self.control.package.update_one({"key": "active"}, {"$set": {"helper": helper}}, True)

    def get_helper(self):
        """Get the active helper package.

        :return: The active helper set for the package.
        :rtype: str
        """
        ret = self.control.package.find_one({"key": "active"})
        # if local compute package used, then 'package' is None
        # if not ret:
        # get framework from round_config instead
        #    ret = self.control.config.find_one({'key': 'round_config'})
        try:
            retcheck = ret["helper"]
            if retcheck == "" or retcheck == " ":  # ugly check for empty string
                return None
            return retcheck
        except (KeyError, IndexError):
            return None

    def list_models(
        self,
        session_id=None,
        limit=None,
        skip=None,
        sort_key="committed_at",
        sort_order=pymongo.DESCENDING,
    ):
        """List all models in the statestore.

        :param session_id: The session id.
        :type session_id: str
        :param limit: The maximum number of models to return.
        :type limit: int
        :param skip: The number of models to skip.
        :type skip: int
        :return: List of models.
        :rtype: list
        """
        result = None

        find_option = {"key": "models"} if session_id is None else {"key": "models", "session_id": session_id}

        projection = {"_id": False, "key": False}

        if limit is not None and skip is not None:
            limit = int(limit)
            skip = int(skip)

            result = self.model.find(find_option, projection).limit(limit).skip(skip).sort(sort_key, sort_order)

        else:
            result = self.model.find(find_option, projection).sort(sort_key, sort_order)

        count = self.model.count_documents(find_option)

        return {
            "result": result,
            "count": count,
        }

    def get_model_trail(self):
        """Get the model trail.

        :return: dictionary of model_id: committed_at
        :rtype: dict
        """
        # TODO Make it so that model order from db is preserved.
        result = self.model.find_one({"key": "model_trail"})
        try:
            if result is not None:
                committed_at = result["committed_at"]
                model = result["model"]
                model_dictionary = dict(zip(model, committed_at))
                return model_dictionary
            else:
                return None
        except (KeyError, IndexError):
            return None

    def get_model_ancestors(self, model_id: str, limit: int):
        """Get the model ancestors.

        :param model_id: The model id.
        :type model_id: str
        :param limit: The maximum number of ancestors to return.
        :type limit: int
        :return: List of model ancestors.
        :rtype: list
        """
        model = self.model.find_one({"key": "models", "model": model_id})
        current_model_id = model["parent_model"] if model is not None else None
        result = []

        for _ in range(limit):
            if current_model_id is None:
                break

            model = self.model.find_one({"key": "models", "model": current_model_id})

            if model is not None:
                result.append(model)
                current_model_id = model["parent_model"]

        return result

    def get_model_descendants(self, model_id: str, limit: int):
        """Get the model descendants.

        :param model_id: The model id.
        :type model_id: str
        :param limit: The maximum number of descendants to return.
        :type limit: int
        :return: List of model descendants.
        :rtype: list
        """
        model: object = self.model.find_one({"key": "models", "model": model_id})
        current_model_id: str = model["model"] if model is not None else None
        result: list = []

        for _ in range(limit):
            if current_model_id is None:
                break

            model: str = self.model.find_one({"key": "models", "parent_model": current_model_id})

            if model is not None:
                result.append(model)
                current_model_id = model["model"]

        result.reverse()

        return result

    def get_model(self, model_id):
        """Get model with id.

        :param model_id: id of model to get
        :type model_id: str
        :return: model with id
        :rtype: ObjectId
        """
        return self.model.find_one({"key": "models", "model": model_id})

    def get_events(self, **kwargs):
        """Get events from the database.

        :param kwargs: query to filter events
        :type kwargs: dict
        :return: events matching query
        :rtype: ObjectId
        """
        # check if kwargs is empty

        result = None
        count = None
        projection = {"_id": False}

        if not kwargs:
            result = self.control.status.find({}, projection).sort("timestamp", pymongo.DESCENDING)
            count = self.control.status.count_documents({})
        else:
            limit = kwargs.pop("limit", None)
            skip = kwargs.pop("skip", None)

            if limit is not None and skip is not None:
                limit = int(limit)
                skip = int(skip)
                result = self.control.status.find(kwargs, projection).sort("timestamp", pymongo.DESCENDING).limit(limit).skip(skip)
            else:
                result = self.control.status.find(kwargs, projection).sort("timestamp", pymongo.DESCENDING)

            count = self.control.status.count_documents(kwargs)

        return {
            "result": result,
            "count": count,
        }

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

    def set_reducer(self, reducer_data):
        """Set the reducer in the statestore.

        :param reducer_data: dictionary of reducer config.
        :type reducer_data: dict
        :return:
        """
        reducer_data["updated_at"] = str(datetime.now())
        self.reducer.update_one({"name": reducer_data["name"]}, {"$set": reducer_data}, True)

    def get_reducer(self):
        """Get reducer.config.

        return: reducer config.
        rtype: ObjectId
        """
        try:
            ret = self.reducer.find_one()
            return ret
        except Exception:
            return None

    def get_combiner(self, name):
        """Get combiner by name.

        :param name: name of combiner to get.
        :type name: str
        :return: The combiner.
        :rtype: ObjectId
        """
        try:
            ret = self.combiners.find_one({"name": name})
            return ret
        except Exception:
            return None

    def get_combiners(self, limit=None, skip=None, sort_key="updated_at", sort_order=pymongo.DESCENDING, projection={}):
        """Get all combiners.

        :param limit: The maximum number of combiners to return.
        :type limit: int
        :param skip: The number of combiners to skip.
        :type skip: int
        :param sort_key: The key to sort by.
        :type sort_key: str
        :param sort_order: The sort order.
        :type sort_order: pymongo.ASCENDING or pymongo.DESCENDING
        :param projection: The projection.
        :type projection: dict
        :return: Dictionary of combiners in result and count.
        :rtype: dict
        """
        result = None
        count = None

        try:
            if limit is not None and skip is not None:
                limit = int(limit)
                skip = int(skip)
                result = self.combiners.find({}, projection).limit(limit).skip(skip).sort(sort_key, sort_order)
            else:
                result = self.combiners.find({}, projection).sort(sort_key, sort_order)

            count = self.combiners.count_documents({})

        except Exception:
            return None

        return {
            "result": result,
            "count": count,
        }

    def set_combiner(self, combiner_data):
        """Set combiner in statestore.

        :param combiner_data: dictionary of combiner config
        :type combiner_data: dict
        :return:
        """
        combiner_data["updated_at"] = str(datetime.now())
        self.combiners.update_one({"name": combiner_data["name"]}, {"$set": combiner_data}, True)

    def delete_combiner(self, combiner):
        """Delete a combiner from statestore.

        :param combiner: name of combiner to delete.
        :type combiner: str
        :return:
        """
        try:
            self.combiners.delete_one({"name": combiner})
        except Exception:
            logger.error(
                "Failed to delete combiner: {}".format(combiner),
            )

    def set_client(self, client_data):
        """Set client in statestore.

        :param client_data: dictionary of client config.
        :type client_data: dict
        :return:
        """
        client_data["updated_at"] = str(datetime.now())
        try:
            # self.clients.update_one({"client_id": client_data["client_id"]}, {"$set": client_data}, True)
            self.clients.update_one(
                {"client_id": client_data["client_id"]},
                {"$set": {k: v for k, v in client_data.items() if v is not None}},
                upsert=True
            )
        except KeyError:
            # If client_id is not present, use name as identifier, for backwards compatibility
            id = str(uuid.uuid4())
            client_data["client_id"] = id
            # self.clients.update_one({"name": client_data["name"]}, {"$set": client_data}, True)
            self.clients.update_one(
                {"client_id": client_data["client_id"]},
                {"$set": {k: v for k, v in client_data.items() if v is not None}},
                upsert=True
            )

    def get_client(self, client_id):
        """Get client by client_id.

        :param client_id: client_id of client to get.
        :type client_id: str
        :return: The client. None if not found.
        :rtype: ObjectId
        """
        try:
            ret = self.clients.find({"key": client_id})
            if list(ret) == []:
                return None
            else:
                return ret
        except Exception:
            return None

    def list_clients(self, limit=None, skip=None, status=None, sort_key="last_seen", sort_order=pymongo.DESCENDING):
        """List all clients registered on the network.

        :param limit: The maximum number of clients to return.
        :type limit: int
        :param skip: The number of clients to skip.
        :type skip: int
        :param status:  online | offline
        :type status: str
        :param sort_key: The key to sort by.
        """
        result = None
        count = None

        try:
            find = {} if status is None else {"status": status}
            projection = {"_id": False, "updated_at": False}

            if limit is not None and skip is not None:
                limit = int(limit)
                skip = int(skip)
                result = self.clients.find(find, projection).limit(limit).skip(skip).sort(sort_key, sort_order)
            else:
                result = self.clients.find(find, projection).sort(sort_key, sort_order)

            count = self.clients.count_documents(find)

        except Exception as e:
            logger.error("{}".format(e))

        return {
            "result": result,
            "count": count,
        }

    def list_combiners_data(self, combiners, sort_key="count", sort_order=pymongo.DESCENDING):
        """List all combiner data.

        :param combiners: list of combiners to get data for.
        :type combiners: list
        :param sort_key: The key to sort by.
        :type sort_key: str
        :param sort_order: The sort order.
        :type sort_order: pymongo.ASCENDING or pymongo.DESCENDING
        :return: list of combiner data.
        :rtype: list(ObjectId)
        """
        result = None

        try:
            pipeline = (
                [
                    {"$match": {"combiner": {"$in": combiners}, "status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$sort": {sort_key: sort_order, "_id": pymongo.ASCENDING}},
                ]
                if combiners is not None
                else [{"$group": {"_id": "$combiner", "count": {"$sum": 1}}}, {"$sort": {sort_key: sort_order, "_id": pymongo.ASCENDING}}]
            )

            result = self.clients.aggregate(pipeline)

        except Exception as e:
            logger.error(e)

        return result

    def report_status(self, msg):
        """Write status message to the database.

        :param msg: The status message.
        :type msg: str
        """
        data = MessageToDict(msg, including_default_value_fields=True)

        if self.status is not None:
            self.status.insert_one(data)

    def report_validation(self, validation):
        """Write model validation to database.

        :param validation: The model validation.
        :type validation: dict
        """
        data = MessageToDict(validation, including_default_value_fields=True)

        if self.validations is not None:
            self.validations.insert_one(data)

    def drop_status(self):
        """Drop the status collection."""
        if self.status:
            self.status.drop()

    def create_session(self, id=None):
        """Create a new session object.

        :param id: The ID of the created session.
        :type id: uuid, str

        """
        if not id:
            id = uuid.uuid4()
        data = {"session_id": str(id)}
        self.sessions.insert_one(data)

    def create_round(self, round_data):
        """Create a new round.

        :param round_data: Dictionary with round data.
        :type round_data: dict
        """
        # TODO: Add check if round_id already exists
        self.rounds.insert_one(round_data)

    def set_session_config(self, id: str, config) -> None:
        """Set the session configuration.

        :param id: The session id
        :type id: str
        :param config: Session configuration
        :type config: dict
        """
        self.sessions.update_one({"session_id": str(id)}, {"$push": {"session_config": config}}, True)

    # Added to accomodate new session config structure
    def set_session_config_v2(self, id: str, config) -> None:
        """Set the session configuration.

        :param id: The session id
        :type id: str
        :param config: Session configuration
        :type config: dict
        """
        self.sessions.update_one({"session_id": str(id)}, {"$set": {"session_config": config}}, True)

    def set_session_status(self, id, status):
        """Set session status.

        :param round_id: The round unique identifier
        :type round_id: str
        :param round_status: The status of the session.
        """
        self.sessions.update_one({"session_id": str(id)}, {"$set": {"status": status}}, True)

    def set_round_combiner_data(self, data):
        """Set combiner round controller data.

        :param data: The combiner data
        :type data: dict
        """
        self.rounds.update_one({"round_id": str(data["round_id"])}, {"$push": {"combiners": data}}, True)

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
