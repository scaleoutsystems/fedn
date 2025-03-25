from abc import abstractmethod
from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto import CombinerDTO
from fedn.network.storage.statestore.stores.sql.shared import CombinerModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class CombinerStore(Store[CombinerDTO]):
    @abstractmethod
    def get_by_name(name: str) -> CombinerDTO:
        pass


class MongoDBCombinerStore(CombinerStore, MongoDBStore[CombinerDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "combiner_id")

    def get_by_name(self, name: str) -> CombinerDTO:
        document = self.database[self.collection].find_one({"name": name})
        if document is None:
            return None
        return self._dto_from_document(document)

    def _document_from_dto(self, item: CombinerDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> CombinerDTO:
        return CombinerDTO().patch_with(from_document(document), throw_on_extra_keys=False)


class SQLCombinerStore(CombinerStore, SQLStore[CombinerDTO, CombinerModel]):
    def __init__(self, Session):
        super().__init__(Session, CombinerModel)

    def update(self, item):
        raise NotImplementedError

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def list(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            entities = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in entities]

    def count(self, **kwargs):
        return self.sql_count(**kwargs)

    def get_by_name(self, name: str) -> CombinerDTO:
        with self.Session() as session:
            entity = session.query(CombinerModel).filter(CombinerModel.name == name).first()
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def _update_orm_model_from_dto(self, entity: CombinerModel, item: CombinerDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("combiner_id", None)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: CombinerModel) -> CombinerDTO:
        orm_dict = from_orm_model(item, CombinerModel)
        orm_dict["combiner_id"] = orm_dict.pop("id")
        return CombinerDTO().populate_with(orm_dict)
