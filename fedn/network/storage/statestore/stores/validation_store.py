from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.validation import ValidationDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import ValidationModel, from_orm_model


class ValidationStore(Store[ValidationDTO]):
    pass


class MongoDBValidationStore(ValidationStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "validation_id")

    def get(self, id: str) -> ValidationDTO:
        item = self.mongo_get(id)
        if item is None:
            return None
        return self._dto_from_document(item)

    def update(self, item: ValidationDTO) -> bool:
        raise NotImplementedError("Update not implemented for ValidationStore")

    def add(self, item: ValidationDTO) -> Tuple[bool, Any]:
        item_dict = item.to_db(exclude_unset=False)
        success, obj = self.mongo_add(item_dict)
        if not success:
            return success, obj
        return success, self._dto_from_document(obj)

    def delete(self, id: str) -> bool:
        return self.mongo_delete(id)

    def select(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> List[ValidationDTO]:
        items = self.mongo_select(limit, skip, sort_key, sort_order, **kwargs)
        return [self._dto_from_document(item) for item in items]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[ValidationDTO]]:
        raise NotImplementedError("List not implemented for ValidationStore")

    def count(self, **kwargs) -> int:
        return self.mongo_count(**kwargs)

    def _dto_from_document(self, document: Dict[str, Any]) -> ValidationDTO:
        dto_dict = from_document(document)
        return ValidationDTO().populate_with(dto_dict)


def translate_key(key: str) -> str:
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


class SQLValidationStore(ValidationStore, SQLStore[ValidationModel]):
    def __init__(self, Session):
        super().__init__(Session, ValidationModel)

    def get(self, id: str) -> ValidationDTO:
        with self.Session() as session:
            item = self.sql_get(session, id)

            if item is None:
                return None

            return self._dto_from_orm_model(item)

    def update(self, item: ValidationDTO) -> bool:
        raise NotImplementedError("Update not implemented for ValidationStore")

    def add(self, item: ValidationDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=False)
            item_dict = self._to_orm_dict(item_dict)
            model = ValidationModel(**item_dict)
            success, obj = self.sql_add(session, model)
            if not success:
                return success, obj
            return success, self._dto_from_orm_model(obj)

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def list(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        raise NotImplementedError("List not implemented for ValidationStore")

    def select(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        kwargs = {translate_key(k): v for k, v in kwargs.items()}
        sort_key = translate_key(sort_key)
        with self.Session() as session:
            items = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in items]

    def count(self, **kwargs):
        kwargs = translate_key(kwargs)
        return self.sql_count(**kwargs)

    def _to_orm_dict(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        item_dict["id"] = item_dict.pop("validation_id")
        sender = item_dict.pop("sender", None)
        if sender:
            item_dict["sender_name"] = sender.get("name")
            item_dict["sender_role"] = sender.get("role")
        receiver = item_dict.pop("receiver", None)
        if receiver:
            item_dict["receiver_name"] = receiver.get("name")
            item_dict["receiver_role"] = receiver.get("role")
        return item_dict

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
