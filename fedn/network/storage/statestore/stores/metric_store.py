from typing import Dict

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.metric import MetricDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import MetricModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class MetricStore(Store[MetricDTO]):
    pass


class MongoDBMetricStore(MetricStore, MongoDBStore[MetricDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "metric_id")

    def _document_from_dto(self, item: MetricDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> MetricDTO:
        item = from_document(document)
        return MetricDTO().patch_with(item, throw_on_extra_keys=False)


def _translate_key_sql(key: str):
    if key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    return key


class SQLMetricStore(MetricStore, SQLStore[MetricDTO, MetricModel]):
    def __init__(self, session):
        super().__init__(session, MetricModel, "metric_id")

    def list(self, limit=0, skip=0, sort_key=None, sort_order=SortOrder.DESCENDING, **kwargs):
        sort_key = _translate_key_sql(sort_key)
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _update_orm_model_from_dto(self, entity: MetricModel, item: MetricDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("metric_id", None)

        sender: Dict = item_dict.pop("sender", {})
        item_dict["sender_name"] = sender.get("name")
        item_dict["sender_role"] = sender.get("role")

        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: MetricModel) -> MetricDTO:
        orm_dict = from_orm_model(item, MetricModel)
        orm_dict["metric_id"] = orm_dict.pop("id")
        orm_dict["sender"] = {
            "name": orm_dict.pop("sender_name"),
            "role": orm_dict.pop("sender_role"),
        }
        return MetricDTO().populate_with(orm_dict)
