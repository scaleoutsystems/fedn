from typing import Any, Dict, List, Tuple, Union

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto import PredictionDTO
from fedn.network.storage.statestore.stores.sql.shared import PredictionModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class PredictionStore(Store[PredictionDTO]):
    pass


def _translate_key_mongo(key: str):
    if key == "correlationId":
        key = "correlation_id"
    elif key == "modelId":
        key = "model_id"
    return key


class MongoDBPredictionStore(PredictionStore, MongoDBStore[PredictionDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "prediction_id")

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[PredictionDTO]:
        entites = super().list(limit, skip, sort_key, sort_order, **kwargs)
        _kwargs = {_translate_key_mongo(k): v for k, v in kwargs.items()}
        _sort_key = _translate_key_mongo(sort_key)
        if _kwargs != kwargs or _sort_key != sort_key:
            entites = super().list(limit, skip, _sort_key, sort_order, **_kwargs) + entites
        return entites

    def count(self, **kwargs) -> int:
        kwargs = {_translate_key_mongo(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _document_from_dto(self, item: PredictionDTO) -> Dict:
        doc = item.to_db()
        return doc

    def _dto_from_document(self, document: Dict) -> PredictionDTO:
        item = from_document(document)

        if "correlationId" in item:
            item["correlation_id"] = item.pop("correlationId")
        if "modelId" in item:
            item["model_id"] = item.pop("modelId")

        pred = PredictionDTO()
        pred.sender.patch_with(item.pop("sender"), throw_on_extra_keys=False)
        pred.receiver.patch_with(item.pop("receiver"), throw_on_extra_keys=False)
        pred.patch_with(item, throw_on_extra_keys=False)
        return pred


def _translate_key(key: str):
    if key == "sender.name":
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
    return key


class SQLPredictionStore(PredictionStore, SQLStore[PredictionModel]):
    def __init__(self, Session):
        super().__init__(Session, PredictionModel)

    def add(self, item: PredictionDTO) -> Tuple[bool, Union[str, PredictionDTO]]:
        with self.Session() as session:
            item_dict = self._to_orm_dict(item)
            prediction = PredictionModel(**item_dict)
            _, obj = self.sql_add(session, prediction)
            return self._dto_from_orm_model(obj)

    def update(self, id: str, item: PredictionDTO) -> bool:
        raise NotImplementedError("Update not implemented for PredictionStore")

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[PredictionDTO]:
        with self.Session() as session:
            kwargs = {_translate_key(k): v for k, v in kwargs.items()}
            sort_key = _translate_key(sort_key)
            entities = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in entities]

    def count(self, **kwargs):
        kwargs = {_translate_key(k): v for k, v in kwargs.items()}
        return self.sql_count(**kwargs)

    def _update_orm_model_from_dto(self, model, item):
        pass

    def _to_orm_dict(self, prediction: PredictionDTO) -> Dict:
        item_dict = prediction.to_db()
        item_dict["id"] = item_dict.pop("prediction_id")
        if "sender" in item_dict:
            sender = item_dict.pop("sender")
            item_dict["sender_name"] = sender.get("name")
            item_dict["sender_role"] = sender.get("role")
        if "receiver" in item_dict:
            receiver = item_dict.pop("receiver")
            item_dict["receiver_name"] = receiver.get("name")
            item_dict["receiver_role"] = receiver.get("role")
        return item_dict

    def _dto_from_orm_model(self, item: PredictionModel) -> PredictionDTO:
        orm_dict = from_orm_model(item, PredictionModel)
        orm_dict["sender"] = {"name": orm_dict.pop("sender_name"), "role": orm_dict.pop("sender_role")}
        orm_dict["receiver"] = {"name": orm_dict.pop("receiver_name"), "role": orm_dict.pop("receiver_role")}
        orm_dict["prediction_id"] = orm_dict.pop("id")
        pred = PredictionDTO()
        pred.populate_with(orm_dict)
        return pred
