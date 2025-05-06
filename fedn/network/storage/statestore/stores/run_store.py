from typing import Dict

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.run import RunDTO
from fedn.network.storage.statestore.stores.sql.shared import RunModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class RunStore(Store[RunDTO]):
    pass


class MongoDBRunStore(RunStore, MongoDBStore[RunDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "run_id")

    def _document_from_dto(self, item: RunDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> RunDTO:
        item = from_document(document)
        return RunDTO().patch_with(item, throw_on_extra_keys=False)

    def update(self, item: RunDTO) -> RunDTO:
        return self.mongo_update(item)


class SQLRunStore(RunStore, SQLStore[RunDTO, RunModel]):
    def __init__(self, session):
        super().__init__(session, RunModel, "run_id")

    def _update_orm_model_from_dto(self, entity: RunModel, item: RunDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("run_id", None)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: RunModel) -> RunDTO:
        orm_dict = from_orm_model(item, RunModel)
        orm_dict["run_id"] = orm_dict.pop("id")
        return RunDTO().populate_with(orm_dict)
