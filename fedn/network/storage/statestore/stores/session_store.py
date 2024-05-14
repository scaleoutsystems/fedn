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
            session_config=data["session_config"] if "session_config" in data else None
        )


class SessionStore(Store[Session]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

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

    def update(self, id: str, item: Session) -> bool:
        raise NotImplementedError("Update not implemented for SessionStore")

    def add(self, item: Session)-> Tuple[bool, Any]:
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

        return {
            "count": response["count"],
            "result": result
        }
