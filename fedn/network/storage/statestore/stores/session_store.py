import datetime
import uuid
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import Store

from .shared import EntityNotFound, from_document


class Session:
    def __init__(self, id: str, session_id: str, status: str, session_config: dict = None):
        self.id = id
        self.session_id = session_id
        self.status = status
        self.session_config = session_config

    def from_dict(data: dict) -> "Session":
        return Session(
            id=str(data["_id"]),
            session_id=data["session_id"] if "session_id" in data else None,
            status=data["status"] if "status" in data else None,
            session_config=data["session_config"] if "session_config" in data else None,
        )


class SessionStore(Store[Session]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def _validate_session_config(self, session_config: dict) -> Tuple[bool, str]:
        if "aggregator" not in session_config:
            return False, "session_config.aggregator is required"

        if "round_timeout" not in session_config:
            return False, "session_config.round_timeout is required"

        if not isinstance(session_config["round_timeout"], (int, float)):
            return False, "session_config.round_timeout must be an integer"

        if "buffer_size" not in session_config:
            return False, "session_config.buffer_size is required"

        if not isinstance(session_config["buffer_size"], int):
            return False, "session_config.buffer_size must be an integer"

        if "model_id" not in session_config or session_config["model_id"] == "":
            return False, "session_config.model_id is required"

        if not isinstance(session_config["model_id"], str):
            return False, "session_config.model_id must be a string"

        if "delete_models_storage" not in session_config:
            return False, "session_config.delete_models_storage is required"

        if not isinstance(session_config["delete_models_storage"], bool):
            return False, "session_config.delete_models_storage must be a boolean"

        if "clients_required" not in session_config:
            return False, "session_config.clients_required is required"

        if not isinstance(session_config["clients_required"], int):
            return False, "session_config.clients_required must be an integer"

        if "validate" not in session_config:
            return False, "session_config.validate is required"

        if not isinstance(session_config["validate"], bool):
            return False, "session_config.validate must be a boolean"

        if "helper_type" not in session_config or session_config["helper_type"] == "":
            return False, "session_config.helper_type is required"

        if not isinstance(session_config["helper_type"], str):
            return False, "session_config.helper_type must be a string"

        return True, ""

    def _validate(self, item: Session) -> Tuple[bool, str]:
        if "session_config" not in item or item["session_config"] is None:
            return False, "session_config is required"

        session_config = None

        if isinstance(item["session_config"], dict):
            session_config = item["session_config"]
        elif isinstance(item["session_config"], list):
            session_config = item["session_config"][0]
        else:
            return False, "session_config must be a dict"

        return self._validate_session_config(session_config)

    def _complement(self, item: Session):
        item["status"] = "Created"
        item["committed_at"] = datetime.datetime.now()

        if "session_id" not in item or item["session_id"] == "" or not isinstance(item["session_id"], str):
            item["session_id"] = str(uuid.uuid4())

    def get(self, id: str, use_typing: bool = False) -> Session:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the session_id (property)
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entity
        """
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            document = self.database[self.collection].find_one({"_id": id_obj})
        else:
            document = self.database[self.collection].find_one({"session_id": id})

        if document is None:
            raise EntityNotFound(f"Entity with (id | session_id) {id} not found")

        return Session.from_dict(document) if use_typing else from_document(document)

    def update(self, id: str, item: Session) -> Tuple[bool, Any]:
        valid, message = self._validate(item)
        if not valid:
            return False, message

        return super().update(id, item)

    def add(self, item: Session) -> Tuple[bool, Any]:
        """Add an entity
        param item: The entity to add
            type: Session
            description: The entity to add
        return: A tuple with a boolean indicating success and the entity
        """
        valid, message = self._validate(item)
        if not valid:
            return False, message

        self._complement(item)

        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for SessionStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Session]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
            description: The maximum number of entities to return
        param skip: The number of entities to skip
            type: int
            description: The number of entities to skip
        param sort_key: The key to sort by
            type: str
            description: The key to sort by
        param sort_order: The order to sort by
            type: pymongo.DESCENDING
            description: The order to sort by
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
            description: Whether to return the entities as typed objects or as dicts.
        param kwargs: Additional query parameters
            type: dict
            description: Additional query parameters
        return: The entities
        """
        response = super().list(limit, skip, sort_key or "session_id", sort_order, use_typing=use_typing, **kwargs)

        result = [Session.from_dict(item) for item in response["result"]] if use_typing else response["result"]

        return {"count": response["count"], "result": result}
