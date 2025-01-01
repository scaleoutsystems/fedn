from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore

from .shared import EntityNotFound, from_document


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


class ClientStore(MongoDBStore[Client]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("client_id", pymongo.DESCENDING)])

    def get(self, id: str) -> Client:
        """Get an entity by id
        param id: The id of the entity
            type: str
        return: The entity
        """
        if ObjectId.is_valid(id):
            response = super().get(id)
        else:
            obj = self._get_client_by_client_id(id)
            response = from_document(obj)
        return response

    def _get_client_by_client_id(self, client_id: str) -> Dict:
        document = self.database[self.collection].find_one({"client_id": client_id})
        if document is None:
            raise EntityNotFound(f"Entity with client_id {client_id} not found")
        return document

    def update(self, by_key: str, value: str, item: Client) -> bool:
        try:
            result = self.database[self.collection].update_one({by_key: value}, {"$set": item})
            if result.modified_count == 1:
                document = self.database[self.collection].find_one({by_key: value})
                return True, from_document(document)
            else:
                return False, "Entity not found"
        except Exception as e:
            return False, str(e)

    def add(self, item: Client) -> Tuple[bool, Any]:
        return super().add(item)

    def upsert(self, item: Client) -> Tuple[bool, Any]:
        try:
            result = self.database[self.collection].update_one(
                {"client_id": item["client_id"]}, {"$set": {k: v for k, v in item.items() if v is not None}}, upsert=True
            )
            id = result.upserted_id
            document = self.database[self.collection].find_one({"_id": id})
            return True, from_document(document)
        except Exception as e:
            return False, str(e)

    def delete(self, id: str) -> bool:
        kwargs = {"_id": ObjectId(id)} if ObjectId.is_valid(id) else {"client_id": id}

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with (id | client_id) {id} not found")

        return super().delete(document["_id"])

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Client]]:
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
        return super().list(limit, skip, sort_key or "last_seen", sort_order, **kwargs)

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
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}},
                ]
                if len(combiners) > 0
                else [
                    {"$match": {"status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}},
                ]
            )

            result = list(self.database[self.collection].aggregate(pipeline))
        except Exception:
            result = {}

        return result
