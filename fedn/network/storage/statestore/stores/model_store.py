from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store

from .shared import EntityNotFound, from_document


class Model:
    def __init__(self, id: str, key: str, model: str, parent_model: str, session_id: str, committed_at: datetime):
        self.id = id
        self.key = key
        self.model = model
        self.parent_model = parent_model
        self.session_id = session_id
        self.committed_at = committed_at


class ModelStore(Store[Model]):
    @abstractmethod
    def list_descendants(self, id: str, limit: int) -> List[Model]:
        pass

    @abstractmethod
    def list_ancestors(self, id: str, limit: int, include_self: bool = False, reverse: bool = False) -> List[Model]:
        pass

    @abstractmethod
    def get_active(self) -> str:
        pass

    @abstractmethod
    def set_active(self, id: str) -> bool:
        pass


class MongoDBModelStore(ModelStore, MongoDBStore[Model]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Model:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        return: The entity
        """
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs["_id"] = id_obj
        else:
            kwargs["model"] = id

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        return from_document(document)

    def _validate(self, item: Model) -> Tuple[bool, str]:
        if "model" not in item or not item["model"]:
            return False, "Model is required"

        return True, ""

    def _complement(self, item: Model):
        if "key" not in item or item["key"] is None:
            item["key"] = "models"

    def update(self, id: str, item: Model) -> Tuple[bool, Any]:
        valid, message = self._validate(item)
        if not valid:
            return False, message

        self._complement(item)

        return super().update(id, item)

    def add(self, item: Model) -> Tuple[bool, Any]:
        valid, message = self._validate(item)
        if not valid:
            return False, message

        self._complement(item)

        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for ModelStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Model]]:
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
        kwargs["key"] = "models"

        return super().list(limit, skip, sort_key or "committed_at", sort_order, **kwargs)

    def list_descendants(self, id: str, limit: int) -> List[Model]:
        """List descendants
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param limit: The maximum number of entities to return
            type: int
        return: A list of entities
        """
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs["_id"] = id_obj
        else:
            kwargs["model"] = id

        model: object = self.database[self.collection].find_one(kwargs)

        if model is None:
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        current_model_id: str = model["model"]
        result: list = []

        for _ in range(limit):
            if current_model_id is None:
                break

            model: str = self.database[self.collection].find_one({"key": "models", "parent_model": current_model_id})

            if model is not None:
                formatted_model = Model.from_dict(model)
                result.append(formatted_model)
                current_model_id = model["model"]
            else:
                break

        result.reverse()

        return result

    def list_ancestors(self, id: str, limit: int, include_self: bool = False, reverse: bool = False) -> List[Model]:
        """List ancestors
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param limit: The maximum number of entities to return
            type: int
        return: A list of entities
        """
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs["_id"] = id_obj
        else:
            kwargs["model"] = id

        model: object = self.database[self.collection].find_one(kwargs)

        if model is None:
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        current_model_id: str = model["parent_model"]
        result: list = []

        if include_self:
            formatted_model = from_document(model)
            result.append(formatted_model)

        for _ in range(limit):
            if current_model_id is None:
                break

            model = self.database[self.collection].find_one({"key": "models", "model": current_model_id})

            if model is not None:
                formatted_model = from_document(model)
                result.append(formatted_model)
                current_model_id = model["parent_model"]
            else:
                break

        if reverse:
            result.reverse()

        return result

    def count(self, **kwargs) -> int:
        """Count entities
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: The count (int)
        """
        kwargs["key"] = "models"
        return super().count(**kwargs)

    def get_active(self) -> str:
        """Get the active model
        return: The active model id (str)
        """
        active_model = self.database[self.collection].find_one({"key": "current_model"})

        if active_model is None:
            raise EntityNotFound("Active model not found")

        return active_model["model"]

    def set_active(self, id: str) -> bool:
        """Set the active model
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        return: True if successful
        """
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs["_id"] = id_obj
        else:
            kwargs["model"] = id

        model = self.database[self.collection].find_one(kwargs)

        if model is None:
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        self.database[self.collection].update_one({"key": "current_model"}, {"$set": {"model": model["model"]}}, upsert=True)

        return True


class SQLModelStore(ModelStore, SQLStore[Model]):
    def __init__(self, database: Database, table: str):
        super().__init__(database, table)

    def create_table(self):
        table_name = super().table_name
        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")

        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id VARCHAR(255) PRIMARY KEY,
            model VARCHAR(255),
            parent_model VARCHAR(255),
            session_id VARCHAR(255),
            committed_at TIMESTAMP
        )
        """
        self.cursor.execute(query)

    def update(self, id, item):
        super().cursor.execute(
            "UPDATE ? SET model = ?, parent_model = ?, session_id = ?, committed_at = ? WHERE id = ?",
            (super().table_name, item.model, item.parent_model, item.session_id, item.committed_at, id),
        )

    def add(self, item):
        super().cursor.execute(
            "INSERT INTO ? (model, parent_model, session_id, committed_at) VALUES (?, ?, ?, ?)",
            (super().table_name, item.model, item.parent_model, item.session_id, item.committed_at),
        )

    def list_descendants(self, id: str, limit: int) -> List[Model]:
        raise NotImplementedError("List descendants not implemented for SQLModelStore")

    def list_ancestors(self, id: str, limit: int, include_self: bool = False, reverse: bool = False) -> List[Model]:
        raise NotImplementedError("List ancestors not implemented for SQLModelStore")

    def get_active(self) -> str:
        raise NotImplementedError("Get active not implemented for SQLModelStore")

    def set_active(self, id: str) -> bool:
        raise NotImplementedError("Set active not implemented for SQLModelStore")
