from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.analytic import AnalyticDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.store import MongoDBStore, Store, from_document


class AnalyticStore(Store[AnalyticDTO]):
    pass


class MongoDBAnalyticStore(AnalyticStore, MongoDBStore[AnalyticDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "id")
        self.database[self.collection].create_index([("sender_id", pymongo.DESCENDING)])

    def add(self, item: AnalyticDTO) -> AnalyticDTO:
        analytic = super().add(item)
        self._delete_old_records(analytic.sender_id)
        return analytic

    def _delete_old_records(self, sender_id: str) -> int:
        time_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)

        result = self.database[self.collection].delete_many({"sender_id": sender_id, "committed_at": {"$lt": time_threshold}})
        return result.deleted_count

    def list(self, limit: int, skip: int, sort_key: str, sort_order=SortOrder.DESCENDING, **kwargs) -> List[AnalyticDTO]:
        return super().list(limit, skip, sort_key or "committed_at", sort_order, **kwargs)

    def _document_from_dto(self, item: AnalyticDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> AnalyticDTO:
        item = from_document(document)
        return AnalyticDTO().patch_with(item, throw_on_extra_keys=False)
