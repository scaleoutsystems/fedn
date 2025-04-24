from typing import Dict

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.status import StatusDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import StatusModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class StatusStore(Store[StatusDTO]):
    pass


class MongoDBStatusStore(StatusStore, MongoDBStore[StatusDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "status_id")

    def _document_from_dto(self, item: StatusDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> StatusDTO:
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


def _translate_key_sql(key: str) -> str:
    if key == "_id":
        key = "id"
    elif key == "logLevel":
        key = "log_level"
    elif key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    return key


class SQLStatusStore(StatusStore, SQLStore[StatusDTO, StatusModel]):
    def __init__(self, Session):
        super().__init__(Session, StatusModel, "status_id")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=SortOrder.DESCENDING, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        sort_key: str = _translate_key_sql(sort_key)
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _update_orm_model_from_dto(self, entity: StatusModel, item: StatusDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("status_id", None)

        sender: Dict = item_dict.pop("sender", {})
        item_dict["sender_name"] = sender.get("name")
        item_dict["sender_role"] = sender.get("role")

        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: StatusModel) -> StatusDTO:
        orm_dict = from_orm_model(item, StatusModel)
        orm_dict["status_id"] = orm_dict.pop("id")
        orm_dict["sender"] = {
            "name": orm_dict.pop("sender_name"),
            "role": orm_dict.pop("sender_role"),
        }
        return StatusDTO().populate_with(orm_dict)
