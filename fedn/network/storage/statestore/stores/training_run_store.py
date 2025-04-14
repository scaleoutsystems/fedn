from typing import Dict

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.training_run import TrainingRunDTO
from fedn.network.storage.statestore.stores.sql.shared import TrainingRunModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class TrainingRunStore(Store[TrainingRunDTO]):
    pass


class MongoDBTrainingRunStore(TrainingRunStore, MongoDBStore[TrainingRunDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "training_run_id")

    def _document_from_dto(self, item: TrainingRunDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> TrainingRunDTO:
        item = from_document(document)
        return TrainingRunDTO().patch_with(item, throw_on_extra_keys=False)

    def update(self, item: TrainingRunDTO) -> TrainingRunDTO:
        return self.mongo_update(item)


class SQLTrainingRunStore(TrainingRunStore, SQLStore[TrainingRunDTO, TrainingRunModel]):
    def __init__(self, session):
        super().__init__(session, TrainingRunModel)

    def _update_orm_model_from_dto(self, entity: TrainingRunModel, item: TrainingRunDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("training_run_id", None)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: TrainingRunModel) -> TrainingRunDTO:
        orm_dict = from_orm_model(item, TrainingRunModel)
        orm_dict["training_run_id"] = orm_dict.pop("id")
        return TrainingRunDTO().populate_with(orm_dict)
