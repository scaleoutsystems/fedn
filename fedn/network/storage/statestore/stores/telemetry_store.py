from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.telemetry import TelemetryDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import TelemetryModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class TelemetryStore(Store[TelemetryDTO]):
    pass


class MongoDBTelemetryStore(TelemetryStore, MongoDBStore[TelemetryDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "telemetry_id")
        self.database[self.collection].create_index([("sender_id", pymongo.DESCENDING)])

    def add(self, item: TelemetryDTO) -> TelemetryDTO:
        telemetry = super().add(item)
        self._delete_old_records(telemetry.sender.client_id, telemetry.key)
        return telemetry

    def _delete_old_records(self, sender_id: str, key: str) -> int:
        time_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)

        result = self.database[self.collection].delete_many({"sender_id": sender_id, "key": key, "committed_at": {"$lt": time_threshold}})
        return result.deleted_count

    def list(self, limit: int, skip: int, sort_key: str, sort_order=SortOrder.DESCENDING, **kwargs) -> List[TelemetryDTO]:
        return super().list(limit, skip, sort_key or "committed_at", sort_order, **kwargs)

    def _document_from_dto(self, item: TelemetryDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> TelemetryDTO:
        item = from_document(document)
        return TelemetryDTO().patch_with(item, throw_on_extra_keys=False)


def _translate_key_sql(key: str):
    if key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    elif key == "sender.client_id":
        key = "sender_client_id"
    return key


class SQLTelemetryStore(TelemetryStore, SQLStore[TelemetryDTO, TelemetryModel]):
    def __init__(self, Session):
        super().__init__(Session, TelemetryModel, "telemetry_id")

    def add(self, item: TelemetryDTO) -> TelemetryDTO:
        telemetry = super().add(item)
        self._delete_old_records(telemetry.sender.client_id, telemetry.key)
        return telemetry

    def _delete_old_records(self, sender_id: str, key: str) -> int:
        with self.Session() as session:
            time_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
            result = (
                session.query(TelemetryModel)
                .filter(TelemetryModel.sender_client_id == sender_id, TelemetryModel.key == key, TelemetryModel.committed_at < time_threshold)
                .delete()
            )
            session.commit()
        return result

    def list(self, limit=0, skip=0, sort_key=None, sort_order=SortOrder.DESCENDING, **kwargs):
        sort_key = _translate_key_sql(sort_key)
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _update_orm_model_from_dto(self, entity: TelemetryModel, item: TelemetryDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("telemetry_id", None)

        sender: Dict = item_dict.pop("sender", {})
        item_dict["sender_name"] = sender.get("name")
        item_dict["sender_role"] = sender.get("role")
        item_dict["sender_client_id"] = sender.get("client_id")

        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: TelemetryModel) -> TelemetryDTO:
        orm_dict = from_orm_model(item, TelemetryModel)
        orm_dict["telemetry_id"] = orm_dict.pop("id")
        orm_dict["sender"] = {
            "name": orm_dict.pop("sender_name"),
            "role": orm_dict.pop("sender_role"),
            "client_id": orm_dict.pop("sender_client_id"),
        }
        return TelemetryDTO().populate_with(orm_dict)
