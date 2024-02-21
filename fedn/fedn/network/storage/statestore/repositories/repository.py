from typing import Dict, Generic, List, TypeVar

import pymongo
from bson import ObjectId
from pymongo.database import Database

from .shared import EntityNotFound, from_document

T = TypeVar('T')


class Repository(Generic[T]):
    def __init__(self, database: Database, collection: str):
        self.database = database
        self.collection = collection

    def get(self, id: str, use_typing: bool = False) -> T:
        """Get an entity by id
        param id: The id of the entity
            type: str
        param use_typing: Whether to return the entity as a typed object or as a dict
        return: The entity
        """
        id_obj = ObjectId(id)
        document = self.database[self.collection].find_one({'_id': id_obj})

        if document is None:
            raise EntityNotFound(f"Entity with id {id} not found")

        return from_document(document) if not use_typing else document

    def update(self, id: str, item: T) -> bool:
        pass

    def add(self, item: T) -> bool:
        pass

    def delete(self, id: str) -> bool:
        pass

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[T]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
        param skip: The number of entities to skip
            type: int
        param sort_key: The key to sort by
            type: str
        param sort_order: The order to sort by
            type: pymongo.DESCENDING | pymongo.ASCENDING
        param use_typing: Whether to return the entities as typed objects or as dicts
            type: bool
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: A dictionary with the count and the result
        """
        cursor = self.database[self.collection].find(kwargs).sort(sort_key, sort_order).skip(skip or 0).limit(limit or 0)

        count = self.database[self.collection].count_documents(kwargs)

        result = [document for document in cursor] if use_typing else [from_document(document) for document in cursor]

        return {
            "count": count,
            "result": result
        }

    def count(self, **kwargs) -> int:
        return self.database[self.collection].count_documents(kwargs)
