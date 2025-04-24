import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Generic, List, Type, TypeVar

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select
from sqlalchemy.orm import Session as SessionClass

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO
from fedn.network.storage.statestore.stores.shared import EntityNotFound, SortOrder
from fedn.network.storage.statestore.stores.sql.shared import MyAbstractBase

MODEL = TypeVar("MODEL", bound=MyAbstractBase)
DTO = TypeVar("DTO", bound=BaseDTO)


def from_document(document: dict) -> dict:
    del document["_id"]
    return document


class Store(ABC, Generic[DTO]):
    """Abstract class for a store.

    OBS! This is an interface, do not add any implementations here.
    """

    @abstractmethod
    def get(self, id: str) -> DTO:
        """Get an entity by id.

        Args:
            id (str): Entity id

        Returns:
            DTO: The entity or null if not found

        """
        pass

    @abstractmethod
    def update(self, item: DTO) -> DTO:
        """Update an existing entity.

        Args:
            item (DTO): The entity to update.

        Returns:
            DTO: The updated entity.

        Raises:
            EntityNotFound: If the entity is not found.
            ValidationError: If validation fails.

        """
        pass

    @abstractmethod
    def add(self, item: DTO) -> DTO:
        """Add an entity.

        Args:
            item (DTO): The entity to update.

        Returns:
            DTO: The updated entity.

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
    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=SortOrder.DESCENDING, **kwargs) -> List[DTO]:
        """Select entities.

        Args:
            limit (int): The maximum number of entities to return
            skip (int): The number of entities to skip
            sort_key (str): The key to sort by
            sort_order (SortOrder.DESCENDING | SortOrder.ASCENDING): The order to sort by
            kwargs (dict): Additional query parameters

        Returns:
            List[DTO]: The list of entities

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


class MongoDBStore(Store[DTO], Generic[DTO]):
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

    def add(self, item: DTO) -> DTO:
        item.check_validity(exclude_primary_id=True)
        item_dict = self._document_from_dto(item)

        if self.primary_key not in item_dict or not item_dict[self.primary_key]:
            item_dict[self.primary_key] = str(uuid.uuid4())
        elif self.database[self.collection].find_one({self.primary_key: item_dict[self.primary_key]}):
            raise Exception(f"Entity with id {item_dict[self.primary_key]} already exists")

        item_dict["committed_at"] = datetime.now()

        self.database[self.collection].insert_one(item_dict)
        document = self.database[self.collection].find_one({self.primary_key: item_dict[self.primary_key]})

        return self._dto_from_document(document)

    def update(self, item: DTO) -> DTO:
        raise NotImplementedError("update not implemented for MongoDBStore by default. Use mongo_update in derived classes.")

    def mongo_update(self, item: DTO) -> DTO:
        item.check_validity()
        item_dict = self._document_from_dto(item)
        id = item_dict[self.primary_key]
        result = self.database[self.collection].update_one({self.primary_key: id}, {"$set": item_dict})
        if result.matched_count == 1:
            document = self.database[self.collection].find_one({self.primary_key: id})
            return self._dto_from_document(document)
        raise EntityNotFound(f"Entity with id {id} not found")

    def delete(self, id: str) -> bool:
        result = self.database[self.collection].delete_one({self.primary_key: id})
        return result.deleted_count == 1

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=SortOrder.DESCENDING, **kwargs) -> List[DTO]:
        _sort_order = sort_order or SortOrder.DESCENDING
        if _sort_order == SortOrder.DESCENDING:
            _sort_order = pymongo.DESCENDING
        elif _sort_order == SortOrder.ASCENDING:
            _sort_order = pymongo.ASCENDING
        else:
            raise ValueError(f"Invalid sort order: {_sort_order}")

        if sort_key and sort_key != "committed_at":
            cursor = self.database[self.collection].find(kwargs).sort({sort_key: _sort_order, "committed_at": _sort_order}).skip(skip or 0).limit(limit or 0)
        else:
            cursor = self.database[self.collection].find(kwargs).sort("committed_at", _sort_order).skip(skip or 0).limit(limit or 0)

        return [self._dto_from_document(document) for document in cursor]

    def count(self, **kwargs) -> int:
        return self.database[self.collection].count_documents(kwargs)

    @abstractmethod
    def _dto_from_document(self, document: Dict) -> DTO:
        pass

    @abstractmethod
    def _document_from_dto(self, item: DTO) -> Dict:
        pass


class SQLStore(Store[DTO], Generic[DTO, MODEL]):
    """Base SQL store implementation."""

    def __init__(self, Session: Type[SessionClass], SQLModel: Type[MODEL], primary_key: str) -> None:
        """Initialize SQLStore."""
        self.SQLModel = SQLModel
        self.Session = Session
        self.primary_key = primary_key

    def get(self, id: str) -> DTO:
        with self.Session() as session:
            stmt = select(self.SQLModel).where(self.SQLModel.id == id)
            entity = session.scalars(stmt).first()
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def add(self, item: DTO) -> DTO:
        item.check_validity(exclude_primary_id=True)
        with self.Session() as session:
            newEntity = self.SQLModel()
            newEntity = self._update_orm_model_from_dto(newEntity, item)

            if not item.primary_id:
                newEntity.id = str(uuid.uuid4())
            else:
                newEntity.id = item.primary_id
            newEntity.committed_at = datetime.now()

            session.add(newEntity)
            session.commit()

            return self._dto_from_orm_model(newEntity)

    def update(self, item: DTO) -> DTO:
        raise NotImplementedError("update not implemented for SQLStore by default. Use sql_update in derived classes.")

    def sql_update(self, item: DTO) -> DTO:
        item.check_validity()
        with self.Session() as session:
            stmt = select(self.SQLModel).where(self.SQLModel.id == item.primary_id)
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                raise EntityNotFound(f"Entity with id {item.primary_id} not found")

            self._update_orm_model_from_dto(existing_item, item)
            session.commit()

            return self._dto_from_orm_model(existing_item)

    def delete(self, id: str) -> bool:
        with self.Session() as session:
            stmt = select(self.SQLModel).where(self.SQLModel.id == id)
            item = session.scalars(stmt).first()

            if item is None:
                return False

            session.delete(item)
            session.commit()

            return True

    def list(self, limit=0, skip=0, sort_key=None, sort_order=SortOrder.DESCENDING, **kwargs) -> List[DTO]:
        with self.Session() as session:
            stmt = select(self.SQLModel)
            if sort_key == self.primary_key:
                sort_key = "id"

            for key, value in kwargs.items():
                if key == self.primary_key:
                    key = "id"
                stmt = stmt.where(getattr(self.SQLModel, key) == value)

            _sort_order = sort_order or SortOrder.DESCENDING
            if _sort_order not in (SortOrder.DESCENDING, SortOrder.ASCENDING):
                raise ValueError(f"Invalid sort order: {_sort_order}")

            secondary_sort_obj = self.SQLModel.__table__.columns.get("committed_at")
            if _sort_order == SortOrder.DESCENDING:
                secondary_sort_obj = secondary_sort_obj.desc()
            if sort_key and sort_key in self.SQLModel.__table__.columns and sort_key != "committed_at":
                sort_obj = self.SQLModel.__table__.columns.get(sort_key)
                if _sort_order == SortOrder.DESCENDING:
                    sort_obj = sort_obj.desc()

                stmt = stmt.order_by(sort_obj, secondary_sort_obj)
            else:
                stmt = stmt.order_by(secondary_sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            elif skip:
                stmt = stmt.offset(skip)

            entities = session.scalars(stmt).all()
            return [self._dto_from_orm_model(entity) for entity in entities]

    def count(self, **kwargs) -> int:
        with self.Session() as session:
            stmt = select(func.count()).select_from(self.SQLModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(self.SQLModel, key) == value)

            return session.scalar(stmt)

    @abstractmethod
    def _dto_from_orm_model(self, item: MODEL) -> DTO:
        pass

    @abstractmethod
    def _update_orm_model_from_dto(self, entity: MODEL, item: DTO) -> MODEL:
        pass
