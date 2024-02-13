from typing import List, Optional

from bson import ObjectId
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
            sender=sender if sender is not None else None
        )


class StatusRepository(Repository[Status]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Status:
        response = super().get(id, use_typing=use_typing)
        return Status.from_dict(response) if use_typing else response

    def put(self, id: str, item: Status) -> bool:
        self.client[self.collection].replace_one({'_id': id}, item.to_dict(), upsert=True)
        return True

    def add(self, item: Status) -> bool:
        self.client[self.collection].insert_one(item.to_dict())
        return True

    def delete(self, id: str) -> bool:
        self.client[self.collection].delete_one({'_id': id})
        return True

    def list(self) -> List[Status]:
        response = super().list()
        return [Status.from_dict(item) for item in response]
