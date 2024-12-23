from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore


class Status:
    def __init__(
        self, id: str, status: str, timestamp: str, log_level: str, data: str, correlation_id: str, type: str, extra: str, session_id: str, sender: dict = None
    ):
        self.id = id
        self.status = status
        self.timestamp = timestamp
        self.log_level = log_level
        self.data = data
        self.correlation_id = correlation_id
        self.type = type
        self.extra = extra
        self.session_id = session_id
        self.sender = sender


class StatusStore(MongoDBStore[Status]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Status:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the status (property)
        return: The entity
        """
        return super().get(id)

    def update(self, id: str, item: Status) -> bool:
        raise NotImplementedError("Update not implemented for StatusStore")

    def add(self, item: Status) -> Tuple[bool, Any]:
        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for StatusStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Status]]:
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
        """
        return super().list(limit, skip, sort_key or "timestamp", sort_order, **kwargs)
