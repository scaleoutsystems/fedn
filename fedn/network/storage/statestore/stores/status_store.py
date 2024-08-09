from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import Store


class Status:
    def __init__(
            self,
            id: str,
            status: str,
            timestamp: str,
            log_level: str,
            data: str,
            correlation_id: str,
            type: str,
            extra: str,
            session_id: str,
            sender: dict = None
    ):
        self.id = id
        self.status = status
        self.timestamp = timestamp
        self.log_level = log_level
        self.data = data
        self.correlation_id = correlation_id
        self.type = type
        self.extra = extra
        self.session_id = session_id
        self.sender = sender

    def from_dict(data: dict) -> "Status":
        return Status(
            id=str(data["_id"]),
            status=data["status"] if "status" in data else None,
            timestamp=data["timestamp"] if "timestamp" in data else None,
            log_level=data["logLevel"] if "logLevel" in data else None,
            data=data["data"] if "data" in data else None,
            correlation_id=data["correlationId"] if "correlationId" in data else None,
            type=data["type"] if "type" in data else None,
            extra=data["extra"] if "extra" in data else None,
            session_id=data["sessionId"] if "sessionId" in data else None,
            sender=data["sender"] if "sender" in data else None
        )


class StatusStore(Store[Status]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Status:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the status (property)
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entity
        """
        response = super().get(id, use_typing=use_typing)
        return Status.from_dict(response) if use_typing else response

    def update(self, id: str, item: Status) -> bool:
        raise NotImplementedError("Update not implemented for StatusStore")

    def add(self, item: Status)-> Tuple[bool, Any]:
        raise NotImplementedError("Add not implemented for StatusStore")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for StatusStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Status]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
            description: The maximum number of entities to return
        param skip: The number of entities to skip
            type: int
            description: The number of entities to skip
        param sort_key: The key to sort by
            type: str
            description: The key to sort by
        param sort_order: The order to sort by
            type: pymongo.DESCENDING
            description: The order to sort by
        param use_typing: Whether to return the entities as typed objects or as dicts
            type: bool
            description: Whether to return the entities as typed objects or as dicts.
        """
        response = super().list(limit, skip, sort_key or "timestamp", sort_order, use_typing=use_typing, **kwargs)

        result = [Status.from_dict(item) for item in response["result"]] if use_typing else response["result"]

        return {"count": response["count"], "result": result}
