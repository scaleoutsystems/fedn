from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.repositories.repository import Repository


class Status:
    def __init__(self, id: str, status: str, timestamp: str, log_level: str, data: str, correlation_id: str, type: str, extra: str, sender: dict = None):
        self.id = id
        self.status = status
        self.timestamp = timestamp
        self.log_level = log_level
        self.data = data
        self.correlation_id = correlation_id
        self.type = type
        self.extra = extra
        self.sender = sender

    def from_dict(data: dict) -> 'Status':
        sender = None
        if 'sender' in data:
            if 'role' in data['sender'] and 'name' in data['sender']:
                sender = data['sender']

        return Status(
            id=str(data['_id']),
            status=data['status'] if 'status' in data else None,
            timestamp=data['timestamp'] if 'timestamp' in data else None,
            log_level=data['logLevel'] if 'logLevel' in data else None,
            data=data['data'] if 'data' in data else None,
            correlation_id=data['correlationId'] if 'correlationId' in data else None,
            type=data['type'] if 'type' in data else None,
            extra=data['extra'] if 'extra' in data else None,
            sender=sender
        )


class StatusRepository(Repository[Status]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Status:
        response = super().get(id, use_typing=use_typing)
        return Status.from_dict(response) if use_typing else response

    def update(self, id: str, item: Status) -> bool:
        raise NotImplementedError("Update not implemented for StatusRepository")

    def add(self, item: Status) -> bool:
        raise NotImplementedError("Add not implemented for StatusRepository")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for StatusRepository")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Status]]:
        response = super().list(limit, skip, sort_key or "timestamp", sort_order, use_typing=use_typing, **kwargs)

        result = [Status.from_dict(item) for item in response['result']] if use_typing else response['result']

        return {'count': response['count'], 'result': result}
