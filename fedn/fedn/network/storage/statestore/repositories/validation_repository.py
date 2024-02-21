from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.repositories.repository import Repository


class Validation:
    def __init__(
            self,
            id: str,
            model_id: str,
            data: str,
            correlation_id: str,
            timestamp: str,
            session_id: str,
            meta: str,
            sender: dict = None,
            receiver: dict = None
    ):
        self.id = id
        self.model_id = model_id
        self.data = data
        self.correlation_id = correlation_id
        self.timestamp = timestamp
        self.session_id = session_id
        self.meta = meta
        self.sender = sender
        self.receiver = receiver

    def from_dict(data: dict) -> 'Validation':
        return Validation(
            id=str(data['_id']),
            model_id=data['modelId'] if 'modelId' in data else None,
            data=data['data'] if 'data' in data else None,
            correlation_id=data['correlationId'] if 'correlationId' in data else None,
            timestamp=data['timestamp'] if 'timestamp' in data else None,
            session_id=data['sessionId'] if 'sessionId' in data else None,
            meta=data['meta'] if 'meta' in data else None,
            sender=data['sender'] if 'sender' in data else None,
            receiver=data['receiver'] if 'receiver' in data else None
        )


class ValidationRepository(Repository[Validation]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Validation:
        response = super().get(id, use_typing=use_typing)
        return Validation.from_dict(response) if use_typing else response

    def update(self, id: str, item: Validation) -> bool:
        raise NotImplementedError("Update not implemented for ValidationRepository")

    def add(self, item: Validation) -> bool:
        raise NotImplementedError("Add not implemented for ValidationRepository")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for ValidationRepository")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Validation]]:
        response = super().list(limit, skip, sort_key or "timestamp", sort_order, use_typing=use_typing, **kwargs)

        result = [Validation.from_dict(item) for item in response['result']] if use_typing else response['result']
        return {
            "count": response['count'],
            "result": result
        }
