from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore


class Validation:
    def __init__(
        self, id: str, model_id: str, data: str, correlation_id: str, timestamp: str, session_id: str, meta: str, sender: dict = None, receiver: dict = None
    ):
        self.id = id
        self.model_id = model_id
        self.data = data
        self.correlation_id = correlation_id
        self.timestamp = timestamp
        self.session_id = session_id
        self.meta = meta
        self.sender = sender
        self.receiver = receiver


class ValidationStore(MongoDBStore[Validation]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Validation:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the validation (property)
        return: The entity
        """
        return super().get(id)

    def update(self, id: str, item: Validation) -> bool:
        raise NotImplementedError("Update not implemented for ValidationStore")

    def add(self, item: Validation) -> Tuple[bool, Any]:
        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for ValidationStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Validation]]:
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
        return: A dictionary with the count and a list of entities
        """
        return super().list(limit, skip, sort_key or "timestamp", sort_order, **kwargs)
