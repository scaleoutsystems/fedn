from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.analytic import AnalyticDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, Store, from_document


class AnalyticStore(Store[AnalyticDTO]):
    pass


def _validate_analytic(analytic: dict) -> Tuple[bool, str]:
    if "sender_id" not in analytic:
        return False, "sender_id is required"
    if "sender_role" not in analytic or analytic["sender_role"] not in ["combiner", "client"]:
        return False, "sender_role must be either 'combiner' or 'client'"
    return analytic, ""


class MongoDBAnalyticStore(AnalyticStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "id")
        self.database[self.collection].create_index([("sender_id", pymongo.DESCENDING)])

    def get(self, id: str) -> AnalyticDTO:
        doc = self.mongo_get(id)
        if doc is None:
            return None
        return self._dto_from_document(doc)

    def update(self, item):
        raise NotImplementedError("Update not implemented for AnalyticStore")

    def _delete_old_records(self, sender_id: str) -> int:
        time_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)

        result = self.database[self.collection].delete_many({"sender_id": sender_id, "committed_at": {"$lt": time_threshold}})
        return result.deleted_count

    def add(self, item: AnalyticDTO) -> Tuple[bool, Any]:
        item_dict = item.to_db(exclude_unset=False)
        valid, msg = _validate_analytic(item_dict)
        if not valid:
            return False, msg

        success, obj = self.mongo_add(item_dict)
        if not success:
            return success, obj

        self._delete_old_records(item_dict["sender_id"])

        return success, self._dto_from_document(obj)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for AnalyticStore")

    def select(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> List[AnalyticDTO]:
        items = self.mongo_select(limit, skip, sort_key or "committed_at", sort_order, **kwargs)
        return [self._dto_from_document(item) for item in items]

    def count(self, **kwargs) -> int:
        return self.mongo_count(**kwargs)

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[AnalyticDTO]]:
        raise NotImplementedError("List not implemented for AnalyticStore")

    def _dto_from_document(self, document: Dict) -> AnalyticDTO:
        item = from_document(document)
        return AnalyticDTO().patch_with(item, throw_on_extra_keys=False)
