import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Tuple, TypeVar

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fedn.network.storage.statestore.stores.shared import from_document

T = TypeVar("T")


class Store(ABC, Generic[T]):
    @abstractmethod
    def get(self, id: str) -> T:
        pass

    @abstractmethod
    def update(self, id: str, item: T) -> Tuple[bool, Any]:
        pass

    @abstractmethod
    def add(self, item: T) -> Tuple[bool, Any]:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def list(
        self,
        limit: int,
        skip: int,
        sort_key: str,
        sort_order=pymongo.DESCENDING,
        **kwargs,
    ) -> Dict[int, List[T]]:
        pass

    @abstractmethod
    def count(self, **kwargs) -> int:
        pass


class MongoDBStore(Store[T], Generic[T]):
    def __init__(self, database: Database, collection: str):
        self.database = database
        self.collection = collection

    def get(self, id: str) -> T:
        """Get an entity by id
        param id: The id of the entity
            type: str
        return: The entity
        """
        if not ObjectId.is_valid(id):
            return None
        id_obj = ObjectId(id)
        document = self.database[self.collection].find_one({"_id": id_obj})
        if document is None:
            return None

        return from_document(document)

    def update(self, id: str, item: T) -> Tuple[bool, Any]:
        try:
            result = self.database[self.collection].update_one({"_id": ObjectId(id)}, {"$set": item})
            if result.modified_count == 1:
                document = self.database[self.collection].find_one({"_id": ObjectId(id)})
                return True, from_document(document)
            else:
                return False, "Entity not found"
        except Exception as e:
            return False, str(e)

    def add(self, item: T) -> Tuple[bool, Any]:
        try:
            result = self.database[self.collection].insert_one(item)
            id = result.inserted_id
            document = self.database[self.collection].find_one({"_id": id})
            return True, from_document(document)
        except Exception as e:
            return False, str(e)

    def delete(self, id: str) -> bool:
        result = self.database[self.collection].delete_one({"_id": ObjectId(id)})
        return result.deleted_count == 1

    def list(
        self,
        limit: int,
        skip: int,
        sort_key: str,
        sort_order=pymongo.DESCENDING,
        **kwargs,
    ) -> Dict[int, List[T]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
        param skip: The number of entities to skip
            type: int
        param sort_key: The key to sort by
            type: str
        param sort_order: The order to sort by
            type: pymongo.DESCENDING | pymongo.ASCENDING
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: A dictionary with the count and the result
        """
        cursor = self.database[self.collection].find(kwargs).sort(sort_key, sort_order).skip(skip or 0).limit(limit or 0)

        count = self.database[self.collection].count_documents(kwargs)

        result = [from_document(document) for document in cursor]

        return {"count": count, "result": result}

    def count(self, **kwargs) -> int:
        """Count entities
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: The count (int)
        """
        return self.database[self.collection].count_documents(kwargs)


class SQLStore(Store[T]):
    def __init__(self, Session):
        self.Session = Session


constraint_naming_conventions = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=constraint_naming_conventions)


class MyAbstractBase(Base):
    __abstract__ = True

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    committed_at: Mapped[datetime] = mapped_column(default=datetime.now())
