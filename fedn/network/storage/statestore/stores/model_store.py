from abc import abstractmethod
from typing import Dict, List

from pymongo.database import Database
from sqlalchemy import select
from sqlalchemy.orm import aliased

from fedn.network.storage.statestore.stores.dto import ModelDTO
from fedn.network.storage.statestore.stores.shared import EntityNotFound, SortOrder
from fedn.network.storage.statestore.stores.sql.shared import ModelModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


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
    def get_leaf_nodes(self) -> List[ModelDTO]:
        """List orphans
        return: A list of entities
        """
        pass


class MongoDBModelStore(ModelStore, MongoDBStore[ModelDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "model")

    def get(self, id: str) -> ModelDTO:
        kwargs = {"key": "models", "model": id}
        document = self.database[self.collection].find_one(kwargs)
        if document is None:
            return None
        return self._dto_from_document(document)

    def update(self, item: ModelDTO) -> ModelDTO:
        item.check_validity()
        item_dict = self._document_from_dto(item)
        id = item_dict[self.primary_key]
        result = self.database[self.collection].update_one({self.primary_key: id, "key": "models"}, {"$set": item_dict})
        if result.matched_count == 1:
            document = self.database[self.collection].find_one({self.primary_key: id, "key": "models"})
            return self._dto_from_document(document)
        raise EntityNotFound(f"Entity with id {id} not found")

    def list(self, limit=0, skip=0, sort_key=None, sort_order=SortOrder.DESCENDING, **kwargs):
        kwargs["key"] = "models"
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs) -> int:
        kwargs["key"] = "models"
        return super().count(**kwargs)

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
                current_model_id = model["model_id"]
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

    def get_leaf_nodes(self) -> List[ModelDTO]:
        parent_ids = self.database[self.collection].distinct("parent_model")

        kwargs = {"key": "models", "model": {"$nin": parent_ids}}
        leaf_nodes = list(self.database[self.collection].find(kwargs))

        result: list = []

        for model in leaf_nodes:
            formatted_model = self._dto_from_document(model)
            result.append(formatted_model)

        return result

    def _document_from_dto(self, item: ModelDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        item_dict["model"] = item_dict.pop("model_id")
        item_dict["key"] = "models"
        return item_dict

    def _dto_from_document(self, document: Dict) -> ModelDTO:
        item_dict = from_document(document)
        item_dict["model_id"] = item_dict.pop("model")
        del item_dict["key"]
        return ModelDTO().patch_with(item_dict, throw_on_extra_keys=False)


class SQLModelStore(ModelStore, SQLStore[ModelDTO, ModelModel]):
    def __init__(self, Session):
        super().__init__(Session, ModelModel, "model_id")

    def update(self, item: ModelDTO) -> ModelDTO:
        return self.sql_update(item)

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

    def get_leaf_nodes(self) -> List[ModelDTO]:
        with self.Session() as session:
            # Define the recursive CTE
            child_alias = aliased(ModelModel)  # Alias for recursion
            query = session.query(ModelModel).outerjoin(child_alias, ModelModel.id == child_alias.parent_id).filter(child_alias.id.is_(None))
            items = session.execute(query).fetchall()

            # Return the list of ancestors
            result = []
            for item in items:
                result.append(self._dto_from_orm_model(item))

            return result

    def _update_orm_model_from_dto(self, entity: ModelModel, item: ModelDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("model_id", None)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: ModelModel) -> ModelDTO:
        orm_dict = from_orm_model(item, ModelModel)
        orm_dict["model_id"] = orm_dict.pop("id")
        del orm_dict["active"]
        return ModelDTO().populate_with(orm_dict)
