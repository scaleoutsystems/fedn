import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select

T = TypeVar("T")


def from_document(document: dict) -> dict:
    del document["_id"]
    return document


class Store(ABC, Generic[T]):
    """Abstract class for a store."""

    @abstractmethod
    def get(self, id: str) -> T:
        """Get an entity by id.

        param id: The id of the entity
            type: str
        return: The entity
        """
        pass

    @abstractmethod
    def update(self, item: T) -> Tuple[bool, Any]:
        """Update an existing entity.

        Will do a patch if fields in T are left unset
        param item: The entity to update
            type: T
        return: A tuple with a boolean and a message if failure or the updated entity if success
        """
        pass

    @abstractmethod
    def add(self, item: T) -> Tuple[bool, Any]:
        """Add an entity.

        param item: The entity to add
              type: T
        return: A tuple with a boolean and a message if failure or the added entity if success
        """
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete an entity.

        param id: The id of the entity
            type: str
        return: A boolean indicating success or failure
        """
        pass

    @abstractmethod
    def select(self, limi: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[T]:
        """List entities.

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
        return: A list of entities
        """
        pass

    @abstractmethod
    def count(self, **kwargs) -> int:
        """Count entities.

        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: The count of entities
        """
        pass


class MongoDBStore:
    """Base MongoDB store implementation."""

    def __init__(self, database: Database, collection: str, primary_key: str) -> None:
        """Initialize MongoDBStore."""
        self.database = database
        self.collection = collection
        self.primary_key = primary_key
        self.database[self.collection].create_index([(self.primary_key, pymongo.DESCENDING)], unique=True)

    def get(self, id: str) -> Dict:
        document = self.database[self.collection].find_one({self.primary_key: id})
        if document is None:
            return None
        return document

    def add(self, item: Dict) -> Tuple[bool, Any]:
        try:
            if self.primary_key not in item:
                item[self.primary_key] = str(uuid.uuid4())
            self.database[self.collection].insert_one(item)
            document = self.database[self.collection].find_one({self.primary_key: item[self.primary_key]})
            return True, document
        except Exception as e:
            return False, str(e)

    def update(self, item: Dict) -> Tuple[bool, Any]:
        try:
            id = item[self.primary_key]
            result = self.database[self.collection].update_one({self.primary_key: id}, {"$set": item})
            if result.modified_count == 1:
                document = self.database[self.collection].find_one({self.primary_key: id})
                return True, document
            return False, "Entity not found"
        except Exception as e:
            return False, str(e)

    def delete(self, id: str) -> bool:
        result = self.database[self.collection].delete_one({self.primary_key: id})
        return result.deleted_count == 1

    def select(
        self,
        limit: int = 0,
        skip: int = 0,
        sort_key: str = None,
        sort_order=pymongo.DESCENDING,
        **kwargs,
    ) -> List[Dict]:
        _sort_key = sort_key or self.primary_key

        cursor = self.database[self.collection].find(kwargs).sort(_sort_key, sort_order).skip(skip or 0).limit(limit or 0)

        return [document for document in cursor]

    def count(self, **kwargs) -> int:
        return self.database[self.collection].count_documents(kwargs)


class SQLStore(Generic[T]):
    """Base SQL store implementation."""

    def __init__(self, Session, primary_key: str, SQLModel: Type[T]) -> None:
        """Initialize SQLStore."""
        self.Session = Session
        self.primary_key = primary_key
        self.SQLModel = SQLModel

    def get(self, session, id: str) -> T:
        stmt = select(self.SQLModel).where(getattr(self.SQLModel, self.primary_key) == id)
        return session.scalars(stmt).first()

    def add(self, session, entity: T) -> Tuple[bool, Any]:
        if not getattr(entity, self.primary_key):
            setattr(entity, self.primary_key, str(uuid.uuid4()))
        session.add(entity)
        session.commit()

        return True, entity

    def update(self, session, item: Dict) -> Tuple[bool, Any]:
        stmt = select(self.SQLModel).where(getattr(self.SQLModel, self.primary_key) == item[self.primary_key])
        existing_item = session.scalars(stmt).first()

        if existing_item is None:
            return False, "Item not found"
        for key, value in item.items():
            setattr(existing_item, key, value)

        session.commit()

        return True, existing_item

    def delete(self, session, id: str) -> bool:
        stmt = select(self.SQLModel).where(getattr(self.SQLModel, self.primary_key) == id)
        item = session.scalars(stmt).first()

        if item is None:
            return False

        session.delete(item)
        session.commit()

        return True

    def select(self, session, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[T]:
        stmt = select(self.SQLModel)

        for key, value in kwargs.items():
            stmt = stmt.where(getattr(self.SQLModel, key) == value)

        _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
        _sort_key: str = sort_key or self.primary_key

        if _sort_key in self.SQLModel.__table__.columns:
            sort_obj = self.SQLModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else self.SQLModel.__table__.columns.get(_sort_key).desc()

            stmt = stmt.order_by(sort_obj)

        if limit:
            stmt = stmt.offset(skip or 0).limit(limit)
        elif skip:
            stmt = stmt.offset(skip)

        return session.scalars(stmt).all()

    def count(self, session, **kwargs) -> int:
        stmt = select(func.count()).select_from(self.SQLModel)

        for key, value in kwargs.items():
            stmt = stmt.where(getattr(self.SQLModel, key) == value)

        return session.scalar(stmt)
