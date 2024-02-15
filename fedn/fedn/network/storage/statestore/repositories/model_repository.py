from datetime import datetime
from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.repositories.repository import Repository


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
        response = super().get(id, use_typing=use_typing)
        return Model.from_dict(response) if use_typing else response

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
