from datetime import datetime
from typing import Dict, List

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.repositories.repository import Repository

from .shared import EntityNotFound, from_document


class Model:
    def __init__(self, id: str, key: str, model: str, parent_model: str, session_id: str, committed_at: datetime):
        self.id = id
        self.key = key
        self.model = model
        self.parent_model = parent_model
        self.session_id = session_id
        self.committed_at = committed_at

    def from_dict(data: dict) -> 'Model':
        return Model(
            id=str(data['_id']),
            key=data['key'] if 'key' in data else None,
            model=data['model'] if 'model' in data else None,
            parent_model=data['parent_model'] if 'parent_model' in data else None,
            session_id=data['session_id'] if 'session_id' in data else None,
            committed_at=data['committed_at'] if 'committed_at' in data else None
        )


class ModelRepository(Repository[Model]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Model:
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs['_id'] = id_obj
        else:
            kwargs['model'] = id

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        return Model.from_dict(document) if use_typing else from_document(document)

    def update(self, id: str, item: Model) -> bool:
        raise NotImplementedError("Update not implemented for ModelRepository")

    def add(self, item: Model) -> bool:
        raise NotImplementedError("Add not implemented for ModelRepository")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for ModelRepository")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Model]]:
        kwargs['key'] = "models"

        response = super().list(limit, skip, sort_key or "committed_at", sort_order, use_typing=use_typing, **kwargs)

        result = [Model.from_dict(item) for item in response['result']] if use_typing else response['result']
        return {
            "count": response['count'],
            "result": result
        }

    def list_descendants(self, id: str, limit: int, use_typing: bool = False) -> List[Model]:
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs['_id'] = id_obj
        else:
            kwargs['model'] = id

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
                formatted_model: Model | dict = Model.from_dict(model) if use_typing else from_document(model)
                result.append(formatted_model)
                current_model_id = model["model"]

        result.reverse()

        return result

    def list_ancestors(self, id: str, limit: int, use_typing: bool = False) -> List[Model]:
        kwargs = {"key": "models"}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs['_id'] = id_obj
        else:
            kwargs['model'] = id

        model: object = self.database[self.collection].find_one(kwargs)

        if model is None:
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        current_model_id: str = model["parent_model"]
        result: list = []

        for _ in range(limit):
            if current_model_id is None:
                break

            model = self.database[self.collection].find_one({"key": "models", "model": current_model_id})

            if model is not None:
                formatted_model: Model | dict = Model.from_dict(model) if use_typing else from_document(model)
                result.append(formatted_model)
                current_model_id = model["parent_model"]

        return result

    def count(self, **kwargs) -> int:
        kwargs['key'] = "models"
        return super().count(**kwargs)
