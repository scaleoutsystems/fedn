from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.validation import ValidationDTO
from fedn.network.storage.statestore.stores.sql.shared import ValidationModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class ValidationStore(Store[ValidationDTO]):
    pass


def translate_key_mongo(key: str):
    if key == "correlation_id":
        key = "correlationId"
    elif key == "model_id":
        key = "modelId"
    elif key == "session_id":
        key = "sessionId"

    return key


class MongoDBValidationStore(ValidationStore, MongoDBStore[ValidationDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "validation_id")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> List[ValidationDTO]:
        items = super().list(limit, skip, sort_key, sort_order, **kwargs)
        _kwargs = {translate_key_mongo(k): v for k, v in kwargs.items()}
        _sort_key = translate_key_mongo(sort_key)
        if _kwargs != kwargs or _sort_key != sort_key:
            items = super().list(limit, skip, _sort_key, sort_order, **_kwargs) + items
        return items

    def count(self, **kwargs) -> int:
        kwargs = {translate_key_mongo(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _document_from_dto(self, item: ValidationDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict[str, Any]) -> ValidationDTO:
        dto_dict = from_document(document)

        if "correlationId" in dto_dict:
            dto_dict["correlation_id"] = dto_dict["correlationId"]
        if "modelId" in dto_dict:
            dto_dict["model_id"] = dto_dict["modelId"]
        if "sessionId" in dto_dict:
            dto_dict["session_id"] = dto_dict["sessionId"]

        return ValidationDTO().patch_with(dto_dict, throw_on_extra_keys=False, verify=True)


def translate_key_sql(key: str) -> str:
    if key == "_id":
        key = "id"
    elif key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    elif key == "receiver.name":
        key = "receiver_name"
    elif key == "receiver.role":
        key = "receiver_role"
    elif key == "correlationId":
        key = "correlation_id"
    elif key == "modelId":
        key = "model_id"
    elif key == "sessionId":
        key = "session_id"

    return key


class SQLValidationStore(ValidationStore, SQLStore[ValidationDTO, ValidationModel]):
    def __init__(self, Session):
        super().__init__(Session, ValidationModel)

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        kwargs = {translate_key_sql(k): v for k, v in kwargs.items()}
        sort_key = translate_key_sql(sort_key)
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = translate_key_sql(kwargs)
        return super().count(**kwargs)

    def _update_orm_model_from_dto(self, entity: ValidationModel, item: ValidationDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("validation_id")

        sender: Dict = item_dict.pop("sender", {})
        item_dict["sender_name"] = sender.get("name")
        item_dict["sender_role"] = sender.get("role")

        receiver: Dict = item_dict.pop("receiver", {})
        item_dict["receiver_name"] = receiver.get("name")
        item_dict["receiver_role"] = receiver.get("role")

        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: ValidationModel) -> ValidationDTO:
        orm_dict = from_orm_model(item, ValidationModel)
        orm_dict["validation_id"] = orm_dict.pop("id")
        sender_name = orm_dict.pop("sender_name")
        sender_role = orm_dict.pop("sender_role")
        if sender_name is not None and sender_role is not None:
            orm_dict["sender"] = {"name": sender_name, "role": sender_role}
        reciever_name = orm_dict.pop("receiver_name")
        receiver_role = orm_dict.pop("receiver_role")
        if reciever_name is not None and receiver_role is not None:
            orm_dict["receiver"] = {"name": reciever_name, "role": receiver_role}

        return ValidationDTO().populate_with(orm_dict)
