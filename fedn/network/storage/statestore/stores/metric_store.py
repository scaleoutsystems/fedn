from typing import Dict

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.metric import MetricDTO
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


class SQLMetricStore(MetricStore, SQLStore[MetricDTO, MetricModel]):
    def __init__(self, session):
        super().__init__(session, MetricModel)

    def _update_orm_model_from_dto(self, entity: MetricModel, item: MetricDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("metric_id", None)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: MetricModel) -> MetricDTO:
        orm_dict = from_orm_model(item, MetricModel)
        orm_dict["metric_id"] = orm_dict.pop("id")
        return MetricDTO().populate_with(orm_dict)
