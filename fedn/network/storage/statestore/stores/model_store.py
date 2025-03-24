from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import select
from sqlalchemy.orm import aliased

from fedn.network.storage.statestore.stores.dto import ModelDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import ModelModel, from_orm_model


class ModelStore(Store[ModelDTO]):
    @abstractmethod
    def list_descendants(self, id: str, limit: int) -> List[ModelDTO]:
        """List descendants
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param limit: The maximum number of entities to return
            type: int
        return: A list of entities
        """
        pass

    @abstractmethod
    def list_ancestors(self, id: str, limit: int, include_self: bool = False, reverse: bool = False) -> List[ModelDTO]:
        """List ancestors
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param limit: The maximum number of entities to return
            type: int
        return: A list of entities
        """
        pass

    @abstractmethod
    def get_active(self) -> str:
        """Get the active model
        return: The active model id (str)
        """
        pass

    @abstractmethod
    def set_active(self, id: str) -> bool:
        """Set the active model
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        return: True if successful
        """
        pass


class MongoDBModelStore(ModelStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "model")

    def get(self, id: str) -> ModelDTO:
        kwargs = {"key": "models", "model": id}
        document = self.database[self.collection].find_one(kwargs)
        if document is None:
            return None

        return self._dto_from_document(document)

    def update(self, item: ModelDTO) -> Tuple[bool, Any]:
        item_dict = item.to_db(exclude_unset=True)
        item_dict = self._complement(item_dict)
        item_dict = self._to_document(item_dict)

        success, obj = self.mongo_update(item_dict)
        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def add(self, item: ModelDTO) -> Tuple[bool, Any]:
        item_dict = item.to_db(exclude_unset=False)
        item_dict = self._complement(item_dict)
        item_dict = self._to_document(item_dict)
        success, obj = self.mongo_add(item_dict)
        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def delete(self, id: str) -> bool:
        return self.mongo_delete(id)

    def select(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        kwargs["key"] = "models"
        entites = self.mongo_select(limit, skip, sort_key, sort_order, **kwargs)
        return [self._dto_from_document(entity) for entity in entites]

    def list_descendants(self, id: str, limit: int) -> List[ModelDTO]:
        kwargs = {"key": "models", "model": id}

        model: object = self.database[self.collection].find_one(kwargs)

        if model is None:
            return None

        current_model_id: str = model["model"]
        result: list = []

        for _ in range(limit):
            if current_model_id is None:
                break

            model: str = self.database[self.collection].find_one({"key": "models", "parent_model": current_model_id})

            if model is not None:
                formatted_model = self._dto_from_document(model)
                result.append(formatted_model)
                current_model_id = model["model"]
            else:
                break

        result.reverse()

        return result

    def list_ancestors(self, id: str, limit: int, include_self: bool = False, reverse: bool = False) -> List[ModelDTO]:
        kwargs = {"key": "models", "model": id}

        model: object = self.database[self.collection].find_one(kwargs)

        if model is None:
            return None

        current_model_id: str = model["parent_model"]
        result: list = []

        if include_self:
            formatted_model = self._dto_from_document(model)
            result.append(formatted_model)

        for _ in range(limit):
            if current_model_id is None:
                break

            model = self.database[self.collection].find_one({"key": "models", "model": current_model_id})

            if model is not None:
                formatted_model = self._dto_from_document(model)
                result.append(formatted_model)
                current_model_id = model["parent_model"]
            else:
                break

        if reverse:
            result.reverse()

        return result

    def count(self, **kwargs) -> int:
        kwargs["key"] = "models"
        return self.mongo_count(**kwargs)

    def get_active(self) -> str:
        active_model = self.database[self.collection].find_one({"key": "current_model"})

        if active_model is None:
            return None

        return active_model["model"]

    def set_active(self, id: str) -> bool:
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs["_id"] = id_obj
        else:
            kwargs["model"] = id

        model = self.database[self.collection].find_one(kwargs)

        if model is None:
            return False

        self.database[self.collection].update_one({"key": "current_model"}, {"$set": {"model": model["model"]}}, upsert=True)

        return True

    def _complement(self, item: Dict) -> Dict:
        if "key" not in item or item["key"] is None:
            item["key"] = "models"
        return item

    def _to_document(self, item_dict: Dict) -> Dict:
        item_dict["model"] = item_dict.pop("model_id")
        return item_dict

    def _dto_from_document(self, document: Dict) -> ModelDTO:
        item_dict = from_document(document)
        item_dict["model_id"] = item_dict.pop("model")
        del item_dict["key"]
        return ModelDTO().patch_with(item_dict, throw_on_extra_keys=False)


class SQLModelStore(ModelStore, SQLStore[ModelDTO]):
    def __init__(self, Session):
        super().__init__(Session, ModelModel)

    def get(self, id: str) -> ModelDTO:
        with self.Session() as session:
            entity = self.sql_get(session, id)
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def update(self, item: ModelDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=True)
            item_dict = self._to_orm_dict(item_dict)
            success, obj = self.sql_update(session, item_dict)
            if success:
                return success, self._dto_from_orm_model(obj)
            return success, obj

    def add(self, item: ModelDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=False)
            item_dict = self._to_orm_dict(item_dict)
            entity = ModelModel(**item_dict)
            success, obj = self.sql_add(session, entity)
            if success:
                return success, self._dto_from_orm_model(obj)
            return success, obj

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def select(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            entities = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(entity) for entity in entities]

    def count(self, **kwargs):
        return self.sql_count(**kwargs)

    def list_descendants(self, id: str, limit: int):
        with self.Session() as session:
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
            for item in items:
                result.append(self._dto_from_orm_model(item))

            return result

    def list_ancestors(self, id: str, limit: int, include_self=False, reverse=False):
        with self.Session() as session:
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
            for item in items:
                result.append(self._dto_from_orm_model(item))

            return result

    def get_active(self) -> str:
        with self.Session() as session:
            active_stmt = select(ModelModel).where(ModelModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                return active_item.id
            return None

    def set_active(self, id: str) -> bool:
        with self.Session() as session:
            stmt = select(ModelModel).where(ModelModel.id == id)
            item = session.scalars(stmt).first()
            if item is None:
                return False

            active_stmt = select(ModelModel).where(ModelModel.active)
            active_items = session.scalars(active_stmt).all()
            for item in active_items:
                item.active = False
            item.active = True
            session.commit()
        return True

    def _to_orm_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("model_id")
        return item_dict

    def _dto_from_orm_model(self, item: ModelModel) -> ModelDTO:
        orm_dict = from_orm_model(item, ModelModel)
        orm_dict["model_id"] = orm_dict.pop("id")
        del orm_dict["active"]
        return ModelDTO().populate_with(orm_dict)
