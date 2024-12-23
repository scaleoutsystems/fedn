from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore

from .shared import EntityNotFound, from_document


class Combiner:
    def __init__(
        self,
        id: str,
        name: str,
        address: str,
        certificate: str,
        config: dict,
        fqdn: str,
        ip: str,
        key: str,
        parent: dict,
        port: int,
        status: str,
        updated_at: str,
    ):
        self.id = id
        self.name = name
        self.address = address
        self.certificate = certificate
        self.config = config
        self.fqdn = fqdn
        self.ip = ip
        self.key = key
        self.parent = parent
        self.port = port
        self.status = status
        self.updated_at = updated_at


class CombinerStore(MongoDBStore[Combiner]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Combiner:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the name (property)
        return: The entity
        """
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            document = self.database[self.collection].find_one({"_id": id_obj})
        else:
            document = self.database[self.collection].find_one({"name": id})

        if document is None:
            raise EntityNotFound(f"Entity with (id | name) {id} not found")

        return from_document(document)

    def update(self, id: str, item: Combiner) -> bool:
        raise NotImplementedError("Update not implemented for CombinerStore")

    def add(self, item: Combiner) -> Tuple[bool, Any]:
        return super().add(item)

    def delete(self, id: str) -> bool:
        if ObjectId.is_valid(id):
            kwargs = {"_id": ObjectId(id)}
        else:
            return False

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with (id) {id} not found")

        return super().delete(document["_id"])

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Combiner]]:
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
        response = super().list(limit, skip, sort_key or "updated_at", sort_order, **kwargs)

        return response

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)
