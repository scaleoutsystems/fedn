from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import Store

from .shared import EntityNotFound, from_document


class Model:
    def __init__(self, id: str, key: str, model: str, parent_model: str, session_id: str, committed_at: datetime):
        self.id = id
        self.key = key
        self.model = model
        self.parent_model = parent_model
        self.session_id = session_id
        self.committed_at = committed_at

    def from_dict(data: dict) -> "Model":
        return Model(
            id=str(data["_id"]),
            key=data["key"] if "key" in data else None,
            model=data["model"] if "model" in data else None,
            parent_model=data["parent_model"] if "parent_model" in data else None,
            session_id=data["session_id"] if "session_id" in data else None,
            committed_at=data["committed_at"] if "committed_at" in data else None,
        )


class ModelStore(Store[Model]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Model:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
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

        return Model.from_dict(document) if use_typing else from_document(document)

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
        raise NotImplementedError("Add not implemented for ModelStore")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for ModelStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Model]]:
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
        kwargs["key"] = "models"

        response = super().list(limit, skip, sort_key or "committed_at", sort_order, use_typing=use_typing, **kwargs)

        result = [Model.from_dict(item) for item in response["result"]] if use_typing else response["result"]
        return {"count": response["count"], "result": result}

    def list_descendants(self, id: str, limit: int, use_typing: bool = False) -> List[Model]:
        """List descendants
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param limit: The maximum number of entities to return
            type: int
        param use_typing: Whether to return the entities as typed objects or as dicts
            type: bool
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
                formatted_model = Model.from_dict(model) if use_typing else from_document(model)
                result.append(formatted_model)
                current_model_id = model["model"]
            else:
                break

        result.reverse()

        return result

    def list_ancestors(self, id: str, limit: int, include_self: bool = False, reverse: bool = False, use_typing: bool = False) -> List[Model]:
        """List ancestors
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param limit: The maximum number of entities to return
            type: int
        param use_typing: Whether to return the entities as typed objects or as dicts
            type: bool
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
            formatted_model = Model.from_dict(model) if use_typing else from_document(model)
            result.append(formatted_model)

        for _ in range(limit):
            if current_model_id is None:
                break

            model = self.database[self.collection].find_one({"key": "models", "model": current_model_id})

            if model is not None:
                formatted_model = Model.from_dict(model) if use_typing else from_document(model)
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

        self.database[self.collection].update_one({"key": "current_model"}, {"$set": {"model": model["model"]}})

        return True
