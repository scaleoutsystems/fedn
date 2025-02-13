from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore, Store


class Analytic:
    def __init__(self, id: str, client_id: str, type: str, execution_duration: int, model_id: str, committed_at: datetime):
        self.id = id
        self.client_id = client_id
        self.type = type
        self.execution_duration = execution_duration
        self.model_id = model_id
        self.committed_at = committed_at


class AnalyticStore(Store[Analytic]):
    pass


def _validate_analytic(analytic: dict) -> Tuple[bool, str]:
    if "client_id" not in analytic:
        return False, "client_id is required"
    if "type" not in analytic or analytic["type"] not in ["training", "inference"]:
        return False, "type must be either 'training' or 'inference'"
    return analytic, ""


def _complete_analytic(analytic: dict) -> dict:
    if "committed_at" not in analytic:
        analytic["committed_at"] = datetime.now()


class MongoDBAnalyticStore(AnalyticStore, MongoDBStore[Analytic]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("client_id", pymongo.DESCENDING)])

    def get(self, id: str) -> Analytic:
        return super().get(id)

    def update(self, id: str, item: Analytic) -> Tuple[bool, Any]:
        pass

    def add(self, item: Analytic) -> Tuple[bool, Any]:
        valid, msg = _validate_analytic(item)
        if not valid:
            return False, msg

        _complete_analytic(item)

        return super().add(item)

    def delete(self, id: str) -> bool:
        pass

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Analytic]]:
        return super().list(limit, skip, sort_key or "committed_at", sort_order, **kwargs)

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)
