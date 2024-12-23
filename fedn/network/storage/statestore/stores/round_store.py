from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore


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
        return super().get(id)

    def update(self, id: str, item: Round) -> bool:
        raise NotImplementedError("Update not implemented for RoundStore")

    def add(self, item: Round) -> Tuple[bool, Any]:
        raise NotImplementedError("Add not implemented for RoundStore")

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
