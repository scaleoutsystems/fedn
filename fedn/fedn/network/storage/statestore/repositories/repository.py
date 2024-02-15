from typing import Dict, Generic, List, TypeVar

import pymongo
from bson import ObjectId
from pymongo.database import Database

from .shared import from_document

T = TypeVar('T')


class Repository(Generic[T]):
    def __init__(self, database: Database, collection: str):
        self.database = database
        self.collection = collection

    def get(self, id: str, use_typing: bool = False) -> T:
        id_obj = ObjectId(id)
        document = self.database[self.collection].find_one({'_id': id_obj})

        if document is None:
            raise KeyError(f"Entity with id {id} not found")

        return from_document(document) if not use_typing else document

    def update(self, id: str, item: T) -> bool:
        pass

    def add(self, item: T) -> bool:
        pass

    def delete(self, id: str) -> bool:
        pass

    def list(self, limit: int | None, skip: int | None, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[T]]:
        cursor = self.database[self.collection].find(kwargs).sort(sort_key, sort_order).skip(skip or 0).limit(limit or 0)

        count = self.database[self.collection].count_documents(kwargs)

        result = [document for document in cursor] if use_typing else [from_document(document) for document in cursor]

        return {
            "count": count,
            "result": result
        }
