from typing import Dict, List

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto import PredictionDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import PredictionModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class PredictionStore(Store[PredictionDTO]):
    pass


class MongoDBPredictionStore(PredictionStore, MongoDBStore[PredictionDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "prediction_id")

    def _document_from_dto(self, item: PredictionDTO) -> Dict:
        doc = item.to_db()
        return doc

    def _dto_from_document(self, document: Dict) -> PredictionDTO:
        item = from_document(document)

        pred = PredictionDTO()
        pred.sender.patch_with(item.pop("sender"), throw_on_extra_keys=False)
        pred.receiver.patch_with(item.pop("receiver"), throw_on_extra_keys=False)
        pred.patch_with(item, throw_on_extra_keys=False)
        return pred


def _translate_key_sql(key: str):
    if key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    elif key == "receiver.name":
        key = "receiver_name"
    elif key == "receiver.role":
        key = "receiver_role"
    return key


class SQLPredictionStore(PredictionStore, SQLStore[PredictionDTO, PredictionModel]):
    def __init__(self, Session):
        super().__init__(Session, PredictionModel, "prediction_id")

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=SortOrder.DESCENDING, **kwargs) -> List[PredictionDTO]:
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        sort_key = _translate_key_sql(sort_key)
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _update_orm_model_from_dto(self, entity: PredictionModel, item: PredictionDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("prediction_id", None)

        if "sender" in item_dict:
            sender: Dict = item_dict.pop("sender")
            item_dict["sender_name"] = sender.get("name")
            item_dict["sender_role"] = sender.get("role")
        if "receiver" in item_dict:
            receiver: Dict = item_dict.pop("receiver")
            item_dict["receiver_name"] = receiver.get("name")
            item_dict["receiver_role"] = receiver.get("role")

        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: PredictionModel) -> PredictionDTO:
        orm_dict = from_orm_model(item, PredictionModel)
        orm_dict["sender"] = {"name": orm_dict.pop("sender_name"), "role": orm_dict.pop("sender_role")}
        orm_dict["receiver"] = {"name": orm_dict.pop("receiver_name"), "role": orm_dict.pop("receiver_role")}
        orm_dict["prediction_id"] = orm_dict.pop("id")
        pred = PredictionDTO()
        pred.populate_with(orm_dict)
        return pred
