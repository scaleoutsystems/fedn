from typing import Any, Dict, List, Tuple, Union

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto import PredictionDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import PredictionModel, from_orm_model


class PredictionStore(Store[PredictionDTO]):
    pass


class MongoDBPredictionStore(PredictionStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "prediction_id")

    def get(self, id: str) -> PredictionDTO:
        entity = self.mongo_get(id)
        if entity is None:
            return None
        return self._dto_from_document(entity)

    def update(self, id: str, item: PredictionDTO) -> bool:
        raise NotImplementedError("Update not implemented for PredictionStore")

    def add(self, item: PredictionDTO) -> Tuple[bool, Any]:
        item_dict = self._document_from_dto(item)
        success, obj = self.mongo_add(item_dict)
        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def delete(self, id: str) -> bool:
        return self.mongo_delete(id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **filter_kwargs) -> List[PredictionDTO]:
        items = self.mongo_select(limit, skip, sort_key, sort_order, **filter_kwargs)
        return [self._dto_from_document(item) for item in items]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[PredictionDTO]]:
        raise NotImplementedError("List not implemented for PredictionStore")

    def count(self, **kwargs) -> int:
        return self.mongo_count(**kwargs)

    def _document_from_dto(self, item: PredictionDTO) -> Dict:
        doc = item.to_db()
        return doc

    def _dto_from_document(self, document: Dict) -> PredictionDTO:
        item = from_document(document)
        pred = PredictionDTO()
        pred.sender.populate_with(item.pop("sender"))
        pred.receiver.populate_with(item.pop("receiver"))
        pred.populate_with(item)
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
    elif key == "correlationId":  # TODO: Why?
        key = "correlation_id"
    elif key == "modelId":  # TODO: Why?
        key = "model_id"
    return key


class SQLPredictionStore(PredictionStore, SQLStore[PredictionModel]):
    def __init__(self, Session):
        super().__init__(Session, PredictionModel)

    def get(self, id: str) -> PredictionDTO:
        with self.Session() as session:
            entity = self.sql_get(session, id)
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def update(self, id: str, item: PredictionDTO) -> bool:
        raise NotImplementedError("Update not implemented for PredictionStore")

    def add(self, item: PredictionDTO) -> Tuple[bool, Union[str, PredictionDTO]]:
        with self.Session() as session:
            item_dict = self._to_orm_dict(item)
            prediction = PredictionModel(**item_dict)
            self.sql_add(session, prediction)
            return True, self._dto_from_orm_model(prediction)

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[PredictionDTO]:
        with self.Session() as session:
            kwargs = {_translate_key(k): v for k, v in kwargs.items()}
            sort_key = _translate_key(sort_key)
            entities = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in entities]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        raise NotImplementedError("List not implemented for PredictionStore")

    def count(self, **kwargs):
        kwargs = {_translate_key(k): v for k, v in kwargs.items()}
        return self.sql_count(**kwargs)

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
