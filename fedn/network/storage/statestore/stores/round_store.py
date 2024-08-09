from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import Store


class Round:
    def __init__(self, id: str, round_id: str, status: str, round_config: dict, combiners: List[dict], round_data: dict):
        self.id = id
        self.round_id = round_id
        self.status = status
        self.round_config = round_config
        self.combiners = combiners
        self.round_data = round_data

    def from_dict(data: dict) -> "Round":
        return Round(
            id=str(data["_id"]),
            round_id=data["round_id"] if "round_id" in data else None,
            status=data["status"] if "status" in data else None,
            round_config=data["round_config"] if "round_config" in data else None,
            combiners=data["combiners"] if "combiners" in data else None,
            round_data=data["round_data"] if "round_data" in data else None
        )


class RoundStore(Store[Round]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Round:
        """Get an entity by id
        param id: The id of the entity
            type: str
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entity
        """
        response = super().get(id, use_typing=use_typing)
        return Round.from_dict(response) if use_typing else response

    def update(self, id: str, item: Round) -> bool:
        raise NotImplementedError("Update not implemented for RoundStore")

    def add(self, item: Round)-> Tuple[bool, Any]:
        raise NotImplementedError("Add not implemented for RoundStore")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for RoundStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Round]]:
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
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entities
        """
        response = super().list(limit, skip, sort_key or "round_id", sort_order, use_typing=use_typing, **kwargs)

        result = [Round.from_dict(item) for item in response["result"]] if use_typing else response["result"]

        return {
            "count": response["count"],
            "result": result
        }
