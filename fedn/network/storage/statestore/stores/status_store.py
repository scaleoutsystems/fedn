from typing import Any, Dict, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.status import StatusDTO
from fedn.network.storage.statestore.stores.sql.shared import StatusModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class StatusStore(Store[StatusDTO]):
    pass


def _translate_key_mongo(key: str) -> str:
    if key == "logLevel":
        key = "log_level"
    elif key == "sender.role":
        key = "sender_role"
    elif key == "sessionId":
        key = "session_id"
    return key


class MongoDBStatusStore(StatusStore, MongoDBStore[StatusDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "status_id")

    def update(self, item: StatusDTO):
        raise NotImplementedError("Update not implemented for StatusStore")

    def delete(self, id: str):
        return self.mongo_delete(id)

    def list(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        items = self.mongo_select(limit, skip, sort_key, sort_order, **kwargs)
        _kwargs = {_translate_key_mongo(k): v for k, v in kwargs.items()}
        _sort_key = _translate_key_mongo(sort_key)
        if _kwargs != kwargs or _sort_key != sort_key:
            items = self.mongo_select(limit, skip, _sort_key, sort_order, **_kwargs) + items
        return [self._dto_from_document(item) for item in items]

    def count(self, **kwargs):
        return self.mongo_count(**kwargs)

    def _document_from_dto(self, item: StatusDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict[str, Any]) -> StatusDTO:
        entity = from_document(document)

        if "logLevel" in entity:
            entity["log_level"] = entity["logLevel"]
        if "correlationId" in entity:
            entity["correlation_id"] = entity["correlationId"]
        if "sessionId" in entity:
            entity["session_id"] = entity["sessionId"]

        if "status_id" not in entity and "id" in entity:
            entity["status_id"] = entity["id"]

        return StatusDTO().patch_with(entity, throw_on_extra_keys=False)


def _translate_key(key: str) -> str:
    if key == "_id":
        key = "id"
    elif key == "logLevel":
        key = "log_level"
    elif key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    elif key == "sessionId":
        key = "session_id"
    return key


class SQLStatusStore(StatusStore, SQLStore[StatusModel]):
    def __init__(self, Session):
        super().__init__(Session, StatusModel)

    def get(self, id: str) -> StatusDTO:
        with self.Session() as session:
            entity = self.sql_get(session, id)
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def update(self, item: StatusDTO):
        raise NotImplementedError

    def add(self, item: StatusDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=False)
            item_dict = self._to_orm_dict(item_dict)
            status = StatusModel(**item_dict)
            success, obj = self.sql_add(session, status)
            return self._dto_from_orm_model(obj)

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            kwargs = {_translate_key(k): v for k, v in kwargs.items()}
            sort_key: str = _translate_key(sort_key)
            items = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in items]

    def count(self, **kwargs):
        kwargs = {_translate_key(k): v for k, v in kwargs.items()}
        return self.sql_count(**kwargs)

    def _to_orm_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("status_id", None)
        sender = item_dict.pop("sender", None)
        if sender:
            item_dict["sender_name"] = sender.get("name")
            item_dict["sender_role"] = sender.get("role")
        return item_dict

    def _dto_from_orm_model(self, item: StatusModel) -> StatusDTO:
        orm_dict = from_orm_model(item, StatusModel)
        orm_dict["status_id"] = orm_dict.pop("id")
        orm_dict["sender"] = {
            "name": orm_dict.pop("sender_name"),
            "role": orm_dict.pop("sender_role"),
        }
        return StatusDTO().populate_with(orm_dict)
