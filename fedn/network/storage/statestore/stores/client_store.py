from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import Store

from .shared import EntityNotFound


class Client:
    def __init__(self, id: str, name: str, combiner: str, combiner_preferred: str, ip: str, status: str, updated_at: str, last_seen: datetime):
        self.id = id
        self.name = name
        self.combiner = combiner
        self.combiner_preferred = combiner_preferred
        self.ip = ip
        self.status = status
        self.updated_at = updated_at
        self.last_seen = last_seen

    def from_dict(data: dict) -> "Client":
        return Client(
            id=str(data["_id"]),
            name=data["name"] if "name" in data else None,
            combiner=data["combiner"] if "combiner" in data else None,
            combiner_preferred=data["combiner_preferred"] if "combiner_preferred" in data else None,
            ip=data["ip"] if "ip" in data else None,
            status=data["status"] if "status" in data else None,
            updated_at=data["updated_at"] if "updated_at" in data else None,
            last_seen=data["last_seen"] if "last_seen" in data else None
        )


class ClientStore(Store[Client]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Client:
        """Get an entity by id
        param id: The id of the entity
            type: str
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entity
        """
        response = super().get(id, use_typing=use_typing)
        return Client.from_dict(response) if use_typing else response

    def update(self, id: str, item: Client) -> bool:
        raise NotImplementedError("Update not implemented for ClientStore")

    def add(self, item: Client)-> Tuple[bool, Any]:
        raise NotImplementedError("Add not implemented for ClientStore")

    def delete(self, id: str) -> bool:
        kwargs = { "_id": ObjectId(id) } if ObjectId.is_valid(id) else { "client_id": id }

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with (id | client_id) {id} not found")

        return super().delete(document["_id"])

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Client]]:
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
        response = super().list(limit, skip, sort_key or "last_seen", sort_order, use_typing=use_typing, **kwargs)

        result = [Client.from_dict(item) for item in response["result"]] if use_typing else response["result"]

        return {
            "count": response["count"],
            "result": result
        }

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)

    def connected_client_count(self, combiners):
        """Count the number of connected clients for each combiner.

        :param combiners: list of combiners to get data for.
        :type combiners: list
        :param sort_key: The key to sort by.
        :type sort_key: str
        :param sort_order: The sort order.
        :type sort_order: pymongo.ASCENDING or pymongo.DESCENDING
        :return: list of combiner data.
        :rtype: list(ObjectId)
        """
        try:
            pipeline = (
                [
                    {"$match": {"combiner": {"$in": combiners}, "status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}}
                ]
                if len(combiners) > 0
                else [
                    {"$match": { "status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}}
                ]
            )

            result = list(self.database[self.collection].aggregate(pipeline))
        except Exception:
            result = {}

        return result
