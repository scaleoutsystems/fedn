from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import MongoDBStore, Store


class Analytic:
    def __init__(self, id: str, sender_id: str, sender_role: str, memory_utilisation: float, cpu_utilisation: float, committed_at: datetime):
        self.id = id
        self.sender_id = sender_id
        self.sender_role = sender_role
        self.memory_utilisation = memory_utilisation
        self.cpu_utilisation = cpu_utilisation
        self.committed_at = committed_at


class AnalyticStore(Store[Analytic]):
    pass


def _validate_analytic(analytic: dict) -> Tuple[bool, str]:
    if "sender_id" not in analytic:
        return False, "sender_id is required"
    if "sender_role" not in analytic or analytic["sender_role"] not in ["combiner", "client"]:
        return False, "sender_role must be either 'combiner' or 'client'"
    return analytic, ""


def _complete_analytic(analytic: dict) -> dict:
    if "committed_at" not in analytic:
        analytic["committed_at"] = datetime.now()


class MongoDBAnalyticStore(AnalyticStore, MongoDBStore[Analytic]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("sender_id", pymongo.DESCENDING)])

    def get(self, id: str) -> Analytic:
        return super().get(id)

    def update(self, id: str, item: Analytic) -> Tuple[bool, Any]:
        pass

    def _delete_old_records(self, sender_id: str) -> int:
        time_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)

        result = self.database[self.collection].delete_many({"sender_id": sender_id, "committed_at": {"$lt": time_threshold}})
        return result.deleted_count

    def add(self, item: Analytic) -> Tuple[bool, Any]:
        valid, msg = _validate_analytic(item)
        if not valid:
            return False, msg

        _complete_analytic(item)

        self._delete_old_records(item["sender_id"])

        return super().add(item)

    def delete(self, id: str) -> bool:
        pass

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Analytic]]:
        return super().list(limit, skip, sort_key or "committed_at", sort_order, **kwargs)

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)
