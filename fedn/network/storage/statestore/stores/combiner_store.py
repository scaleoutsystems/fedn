from abc import abstractmethod
from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto import CombinerDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import CombinerModel, from_orm_model


class CombinerStore(Store[CombinerDTO]):
    @abstractmethod
    def get_by_name(name: str) -> CombinerDTO:
        pass


class MongoDBCombinerStore(CombinerStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "combiner_id")

    def get(self, id: str) -> CombinerDTO:
        obj = self.mongo_get(id)
        if obj is None:
            return None
        return self._dto_from_document(obj)

    def update(self, item: CombinerDTO):
        raise NotImplementedError("Update not implemented for CombinerStore")

    def add(self, item: CombinerDTO):
        item_dict = item.to_db(exclude_unset=False)
        success, obj = self.mongo_add(item_dict)
        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def delete(self, id: str) -> bool:
        return self.mongo_delete(id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **filter_kwargs) -> List[CombinerDTO]:
        entities = self.mongo_select(limit, skip, sort_key, sort_order, **filter_kwargs)
        result = []
        for entity in entities:
            result.append(self._dto_from_document(entity))
        return result

    def count(self, **kwargs) -> int:
        return self.mongo_count(**kwargs)

    def get_by_name(self, name: str) -> CombinerDTO:
        document = self.database[self.collection].find_one({"name": name})
        if document is None:
            return None
        return self._dto_from_document(document)

    def _dto_from_document(self, document: Dict) -> CombinerDTO:
        return CombinerDTO().populate_with(from_document(document))


class SQLCombinerStore(CombinerStore, SQLStore[CombinerDTO]):
    def __init__(self, Session):
        super().__init__(Session, CombinerModel)

    def get(self, id: str) -> CombinerDTO:
        with self.Session() as session:
            entity = self.sql_get(session, id)
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def update(self, item):
        raise NotImplementedError

    def add(self, item):
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=False)
            item_dict = self._to_orm_dict(item_dict)
            entity = CombinerModel(**item_dict)
            success, obj = self.sql_add(session, entity)
            if success:
                return success, self._dto_from_orm_model(obj)
            return success, obj

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def select(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
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

    def _to_orm_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("combiner_id")
        return item_dict

    def _dto_from_orm_model(self, item: CombinerModel) -> CombinerDTO:
        orm_dict = from_orm_model(item, CombinerModel)
        orm_dict["combiner_id"] = orm_dict.pop("id")
        return CombinerDTO().populate_with(orm_dict)
