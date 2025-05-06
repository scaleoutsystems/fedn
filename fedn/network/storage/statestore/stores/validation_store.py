from typing import Dict

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.validation import ValidationDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import ValidationModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class ValidationStore(Store[ValidationDTO]):
    pass


class MongoDBValidationStore(ValidationStore, MongoDBStore[ValidationDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "validation_id")

    def _document_from_dto(self, item: ValidationDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> ValidationDTO:
        dto_dict = from_document(document)

        return ValidationDTO().patch_with(dto_dict, throw_on_extra_keys=False, verify=True)


def _translate_key_sql(key: str) -> str:
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

    return key


class SQLValidationStore(ValidationStore, SQLStore[ValidationDTO, ValidationModel]):
    def __init__(self, Session):
        super().__init__(Session, ValidationModel, "validation_id")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=SortOrder.DESCENDING, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        sort_key = _translate_key_sql(sort_key)
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = _translate_key_sql(kwargs)
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
