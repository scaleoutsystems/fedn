import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, Union

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select
from sqlalchemy.orm import Session as SessionClass

from fedn.network.storage.statestore.stores.shared import EntityNotFound

T = TypeVar("T")


def from_document(document: dict) -> dict:
    del document["_id"]
    return document


class Store(ABC, Generic[T]):
    """Abstract class for a store."""

    @abstractmethod
    def get(self, id: str) -> T:
        """Get an entity by id.

        Args:
            id (str): Entity id

        Returns:
            T: The entity or null if not found

        """
        pass

    @abstractmethod
    def update(self, item: T) -> T:
        """Update an existing entity.

        Args:
            item (T): The entity to update.

        Returns:
            T: The updated entity.

        Raises:
            EntityNotFound: If the entity is not found.
            ValidationError: If validation fails.

        """
        pass

    @abstractmethod
    def add(self, item: T) -> T:
        """Add an entity.

        Args:
            item (T): The entity to update.

        Returns:
            T: The updated entity.

        Raises:
            ValidationError: If validation fails.

        """
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete an entity.

        Args:
          id (str): The id of the entity

        Returns:
            Bool: success or failure

        """
        pass

    @abstractmethod
    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[T]:
        """Select entities.

        Args:
            limit (int): The maximum number of entities to return
            skip (int): The number of entities to skip
            sort_key (str): The key to sort by
            sort_order (pymongo.DESCENDING | pymongo.ASCENDING): The order to sort by
            kwargs (dict): Additional query parameters

        Returns:
            List[T]: The list of entities

        """
        pass

    @abstractmethod
    def count(self, **kwargs) -> int:
        """Count entities.


        Args:
            kwargs (dict): Additional query parameters, example: {"key": "models"}

        Returns:
            int: The number of entities

        """
        pass


class MongoDBStore(Store[T], Generic[T]):
    """Base MongoDB store implementation."""

    def __init__(self, database: Database, collection: str, primary_key: str) -> None:
        """Initialize MongoDBStore."""
        self.database = database
        self.collection = collection
        self.primary_key = primary_key
        self.database[self.collection].create_index([(self.primary_key, pymongo.DESCENDING)])

    def get(self, id):
        document = self.database[self.collection].find_one({self.primary_key: id})
        if document is None:
            return None
        return self._dto_from_document(document)

    def add(self, item: T) -> T:
        item_dict = self._document_from_dto(item)

        if self.primary_key not in item_dict or not item_dict[self.primary_key]:
            item_dict[self.primary_key] = str(uuid.uuid4())
        item_dict["committed_at"] = datetime.now()

        self.database[self.collection].insert_one(item_dict)
        document = self.database[self.collection].find_one({self.primary_key: item_dict[self.primary_key]})

        return self._dto_from_document(document)

    def update(self, item: T) -> T:
        raise NotImplementedError("update not implemented for MongoDBStore by default. Use mongo_update in derived classes.")

    def mongo_update(self, item: T) -> T:
        item_dict = self._document_from_dto(item)
        id = item_dict[self.primary_key]
        result = self.database[self.collection].update_one({self.primary_key: id}, {"$set": item_dict})
        if result.modified_count == 1:
            document = self.database[self.collection].find_one({self.primary_key: id})
            return self._dto_from_document(document)
        raise EntityNotFound(f"Entity with id {id} not found")

    def delete(self, id: str) -> bool:
        result = self.database[self.collection].delete_one({self.primary_key: id})
        return result.deleted_count == 1

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[T]:
        _sort_order = sort_order or pymongo.DESCENDING
        if sort_key and sort_key != self.primary_key:
            cursor = (
                self.database[self.collection]
                .find(kwargs)
                .sort({sort_key: _sort_order, self.primary_key: pymongo.DESCENDING})
                .skip(skip or 0)
                .limit(limit or 0)
            )
        else:
            cursor = self.database[self.collection].find(kwargs).sort(self.primary_key, pymongo.DESCENDING).skip(skip or 0).limit(limit or 0)

        return [self._dto_from_document(document) for document in cursor]

    def count(self, **kwargs) -> int:
        return self.database[self.collection].count_documents(kwargs)

    @abstractmethod
    def _dto_from_document(self, document: Dict) -> T:
        pass

    @abstractmethod
    def _document_from_dto(self, item: T) -> Dict:
        pass


class SQLStore(Store[T], Generic[T]):
    """Base SQL store implementation."""

    def __init__(self, Session: Type[SessionClass], SQLModel: Type[T]) -> None:
        """Initialize SQLStore."""
        self.SQLModel = SQLModel
        self.Session = Session

    def get(self, id: str) -> T:
        with self.Session() as session:
            stmt = select(self.SQLModel).where(self.SQLModel.id == id)
            entity = session.scalars(stmt).first()
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def sql_add(self, session, entity: T) -> Tuple[bool, Any]:
        if not entity.id:
            entity.id = str(uuid.uuid4())
        session.add(entity)
        session.commit()

        return True, entity

    def sql_update(self, session: SessionClass, item: Dict) -> Tuple[bool, Any]:
        stmt = select(self.SQLModel).where(self.SQLModel.id == item["id"])
        existing_item = session.scalars(stmt).first()

        if existing_item is None:
            return False, "Item not found"
        for key, value in item.items():
            setattr(existing_item, key, value)

        session.commit()

        return True, existing_item

    def sql_delete(self, id: str) -> bool:
        with self.Session() as session:
            stmt = select(self.SQLModel).where(self.SQLModel.id == id)
            item = session.scalars(stmt).first()

            if item is None:
                return False

            session.delete(item)
            session.commit()

            return True

    def sql_select(self, session: SessionClass, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[T]:
        stmt = select(self.SQLModel)

        for key, value in kwargs.items():
            stmt = stmt.where(getattr(self.SQLModel, key) == value)

        _sort_order = sort_order or pymongo.DESCENDING

        secondary_sort_obj = self.SQLModel.__table__.columns.get("id").desc()
        if sort_key and sort_key in self.SQLModel.__table__.columns:
            sort_obj = self.SQLModel.__table__.columns.get(sort_key)
            if _sort_order == pymongo.DESCENDING:
                sort_obj = sort_obj.desc()

            stmt = stmt.order_by(sort_obj, secondary_sort_obj)
        else:
            stmt = stmt.order_by(secondary_sort_obj)

        if limit:
            stmt = stmt.offset(skip or 0).limit(limit)
        elif skip:
            stmt = stmt.offset(skip)

        return session.scalars(stmt).all()

    def sql_count(self, **kwargs) -> int:
        with self.Session() as session:
            stmt = select(func.count()).select_from(self.SQLModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(self.SQLModel, key) == value)

            return session.scalar(stmt)

    @abstractmethod
    def _dto_from_orm_model(self, item: T) -> T:
        pass

    @abstractmethod
    def _update_orm_model_from_dto(self, model, item: T):
        pass
