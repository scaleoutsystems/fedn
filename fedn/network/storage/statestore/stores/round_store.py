from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore

from .shared import EntityNotFound, from_document


class Round:
    def __init__(self, id: str, round_id: str, status: str, round_config: dict, combiners: List[dict], round_data: dict):
        self.id = id
        self.round_id = round_id
        self.status = status
        self.round_config = round_config
        self.combiners = combiners
        self.round_data = round_data


class RoundStore(MongoDBStore[Round]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Round:
        """Get an entity by id
        param id: The id of the entity
            type: str
        return: The entity
        """
        kwargs = {}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs["_id"] = id_obj
        else:
            kwargs["round_id"] = id

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        return from_document(document)

    def update(self, id: str, item: Round) -> bool:
        return super().update(id, item)

    def add(self, item: Round) -> Tuple[bool, Any]:
        round_id = item["round_id"]
        existing = self.database[self.collection].find_one({"round_id": round_id})

        if existing is not None:
            return False, "Round with round_id already exists"

        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for RoundStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Round]]:
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
        return: The entities
        """
        return super().list(limit, skip, sort_key or "round_id", sort_order, **kwargs)
        return super().list(limit, skip, sort_key or "round_id", sort_order, **kwargs)
