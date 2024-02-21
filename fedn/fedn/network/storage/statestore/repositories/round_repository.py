from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.repositories.repository import Repository


class Round:
    def __init__(self, id: str, round_id: str, status: str, round_config: dict, combiners: List[dict], round_data: dict):
        self.id = id
        self.round_id = round_id
        self.status = status
        self.round_config = round_config
        self.combiners = combiners
        self.round_data = round_data

    def from_dict(data: dict) -> 'Round':
        return Round(
            id=str(data['_id']),
            round_id=data['round_id'] if 'round_id' in data else None,
            status=data['status'] if 'status' in data else None,
            round_config=data['round_config'] if 'round_config' in data else None,
            combiners=data['combiners'] if 'combiners' in data else None,
            round_data=data['round_data'] if 'round_data' in data else None
        )


class RoundRepository(Repository[Round]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Round:
        response = super().get(id, use_typing=use_typing)
        return Round.from_dict(response) if use_typing else response

    def update(self, id: str, item: Round) -> bool:
        raise NotImplementedError("Update not implemented for RoundRepository")

    def add(self, item: Round) -> bool:
        raise NotImplementedError("Add not implemented for RoundRepository")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for RoundRepository")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Round]]:
        response = super().list(limit, skip, sort_key or "round_id", sort_order, use_typing=use_typing, **kwargs)

        result = [Round.from_dict(item) for item in response['result']] if use_typing else response['result']

        return {
            "count": response["count"],
            "result": result
        }