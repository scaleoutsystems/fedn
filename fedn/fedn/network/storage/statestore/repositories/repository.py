from typing import Generic, List, TypeVar

from bson import ObjectId
from pymongo.database import Database

T = TypeVar('T')


def from_document(document: dict) -> dict:
    document['id'] = str(document['_id'])
    del document['_id']
    return document


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

    def put(self, id: str, item: T) -> bool:
        pass

    def add(self, item: T) -> bool:
        pass

    def delete(self, id: str) -> bool:
        pass

    def list(self, use_typing: bool = False) -> List[T]:
        pass
