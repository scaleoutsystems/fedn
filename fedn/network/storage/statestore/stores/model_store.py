from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import func, select
from sqlalchemy.orm import aliased
from sqlalchemy.sql import text

from fedn.network.storage.statestore.stores.shared import EntityNotFound, from_document
from fedn.network.storage.statestore.stores.sql_models import ModelModel
from fedn.network.storage.statestore.stores.store import MongoDBStore, Session, SQLStore, Store


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


def validate(item: Model) -> Tuple[bool, str]:
    if "model" not in item or not item["model"]:
        return False, "Model is required"

    return True, ""


class MongoDBModelStore(ModelStore, MongoDBStore[Model]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("model", pymongo.DESCENDING)])

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

    def _complement(self, item: Model):
        if "key" not in item or item["key"] is None:
            item["key"] = "models"

    def update(self, id: str, item: Model) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message

        self._complement(item)

        return super().update(id, item)

    def add(self, item: Model) -> Tuple[bool, Any]:
        valid, message = validate(item)
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


def from_row(row: ModelModel) -> Model:
    return {
        "id": row.id,
        "model": row.id,
        "committed_at": row.committed_at,
        "parent_model": row.parent_model,
        "session_id": row.session_id,
        "name": row.name,
        "active": row.active,
    }


class SQLModelStore(ModelStore, SQLStore[Model]):
    def get(self, id: str) -> Model:
        with Session() as session:
            stmt = select(ModelModel).where(ModelModel.id == id)
            item = session.scalars(stmt).first()
            if item is None:
                raise EntityNotFound("Entity not found")
            return from_row(item)

    def update(self, id: str, item: Model) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message
        with Session() as session:
            stmt = select(ModelModel).where(ModelModel.id == id)
            existing_item = session.execute(stmt).first()
            if existing_item is None:
                raise EntityNotFound(f"Entity not found {id}")

            existing_item.parent_model = item["parent_model"]
            existing_item.name = item["name"]
            existing_item.session_id = item.get("session_id")
            existing_item.committed_at = item["committed_at"]
            existing_item.active = item["active"]

            return True, from_row(existing_item)

    def add(self, item: Model) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message

        with Session() as session:
            id: str = None
            if "model" in item:
                id = item["model"]
            elif "id" in item:
                id = item["id"]

            item = ModelModel(
                id=id,
                parent_model=item["parent_model"],
                name=item["name"],
                session_id=item.get("session_id"),
                committed_at=item["committed_at"],
                active=item["active"] if "active" in item else False,
            )
            session.add(item)
            session.commit()
            return True, from_row(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with Session() as session:
            stmt = select(ModelModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(ModelModel, key) == value)

            _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
            _sort_key: str = sort_key or "committed_at"

            stmt = stmt.order_by(text(f"{_sort_key} {_sort_order}"))

            if limit != 0:
                stmt = stmt.offset(skip or 0).limit(limit)

            items = session.scalars(stmt).all()

            result = []
            for i in items:
                result.append(from_row(i))

            count = session.scalar(select(func.count()).select_from(ModelModel))

            return {"count": count, "result": result}

    def count(self, **kwargs):
        with Session() as session:
            stmt = select(func.count()).select_from(ModelModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(ModelModel, key) == value)

            count = session.scalar(stmt)

            return count

    def list_descendants(self, id: str, limit: int):
        with Session() as session:
            # Define the recursive CTE
            descendant = aliased(ModelModel)  # Alias for recursion
            cte = select(ModelModel).where(ModelModel.parent_model == id).cte(name="descendant_cte", recursive=True)
            cte = cte.union_all(select(descendant).where(descendant.parent_model == cte.c.id))

            # Final query with optional limit
            query = select(cte)
            if limit is not None:
                query = query.limit(limit)

            # Execute the query
            items = session.execute(query).fetchall()

            # Return the list of descendants
            result = []
            for i in items:
                result.append(from_row(i))

            return result

    def list_ancestors(self, id: str, limit: int, include_self=False, reverse=False):
        with Session() as session:
            # Define the recursive CTE
            ancestor = aliased(ModelModel)  # Alias for recursion
            cte = select(ModelModel).where(ModelModel.id == id).cte(name="ancestor_cte", recursive=True)
            cte = cte.union_all(select(ancestor).where(ancestor.id == cte.c.parent_model))

            # Final query with optional limit
            query = select(cte)
            if limit is not None:
                query = query.limit(limit)

            # Execute the query
            items = session.execute(query).fetchall()

            # Return the list of ancestors
            result = []
            for i in items:
                result.append(from_row(i))

            return result

    def get_active(self) -> str:
        with Session() as session:
            active_stmt = select(ModelModel).where(ModelModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                return active_item.id
            raise EntityNotFound("Entity not found")

    def set_active(self, id: str) -> bool:
        with Session() as session:
            active_stmt = select(ModelModel).where(ModelModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                active_item.active = False

            stmt = select(ModelModel).where(ModelModel.id == id)
            item = session.scalars(stmt).first()

            if item is None:
                raise EntityNotFound("Entity not found")

            item.active = True
            session.commit()
        return True
