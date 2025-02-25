from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select

from fedn.network.storage.statestore.stores.dto import ClientDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import ClientModel, from_orm_model


class ClientStore(Store[ClientDTO]):
    """Client store interface."""

    @abstractmethod
    def connected_client_count(self, combiners: List[str]) -> List[ClientDTO]:
        """Count the number of connected clients for each combiner.

        :param combiners: list of combiners to get data for.
        :type combiners: list
        :param sort_key: The key to sort by.
        :type sort_key: str
        :param sort_order: The sort order.
        :type sort_order: pymongo.ASCENDING or pymongo.DESCENDING
        :return: list of combiner data.
        :rtype: list(ObjectId)
        """
        pass


class MongoDBClientStore(ClientStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "client_id")

    def get(self, client_id: str) -> ClientDTO:
        entity = MongoDBStore.get(self, client_id)
        if entity is None:
            return None
        return self._dto_from_document(entity)

    def add(self, item: ClientDTO) -> Tuple[bool, Any]:
        item_dict = item.to_dict(exclude_unset=False)
        success, obj = MongoDBStore.add(self, item_dict)
        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def update(self, item: ClientDTO) -> Tuple[bool, Any]:
        item_dict = item.to_dict()
        success, obj = MongoDBStore.update(self, item_dict)
        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def delete(self, client_id: str) -> bool:
        return MongoDBStore.delete(self, client_id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **filter_kwargs) -> List[ClientDTO]:
        entites = MongoDBStore.select(self, limit, skip, sort_key, sort_order, **filter_kwargs)
        return [self._dto_from_document(entity) for entity in entites]

    def count(self, **kwargs) -> int:
        return MongoDBStore.count(self, **kwargs)

    def connected_client_count(self, combiners: List[str]) -> List:
        try:
            pipeline = (
                [
                    {"$match": {"combiner": {"$in": combiners}, "status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}},
                ]
                if len(combiners) > 0
                else [
                    {"$match": {"status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}},
                ]
            )

            result = list(self.database[self.collection].aggregate(pipeline))
        except Exception:
            result = []

        return result

    def _dto_from_document(self, document: Dict) -> ClientDTO:
        return ClientDTO().populate_with(from_document(document))


class SQLClientStore(ClientStore):
    def __init__(self, Session):
        self.Session = Session
        self.sql_helper = SQLStore[ClientModel](ClientModel)

    def get(self, id: str) -> ClientDTO:
        with self.Session() as session:
            entity = self.sql_helper.get(session, id)
            if entity is None:
                return None
            return self._dto_from_orm_model(entity)

    def add(self, item: ClientDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_dict(exclude_unset=False)
            item_dict = self._to_orm_dict(item_dict)
            entity = ClientModel(**item_dict)
            success, obj = self.sql_helper.add(session, entity)
            if success:
                return success, self._dto_from_orm_model(obj)
            return success, obj

    def update(self, item: ClientDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_dict(exclude_unset=True)
            item_dict = self._to_orm_dict(item_dict)
            success, obj = self.sql_helper.update(session, item_dict)
            if success:
                return success, self._dto_from_orm_model(obj)
            return success, obj

    def delete(self, id) -> bool:
        with self.Session() as session:
            return self.sql_helper.delete(session, id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[ClientDTO]:
        with self.Session() as session:
            entities = self.sql_helper.select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in entities]

    def count(self, **kwargs):
        with self.Session() as session:
            return self.sql_helper.count(session, **kwargs)

    def connected_client_count(self, combiners) -> List[Dict]:
        with self.Session() as session:
            stmt = select(ClientModel.combiner, func.count(ClientModel.combiner)).group_by(ClientModel.combiner)
            if combiners:
                stmt = stmt.where(ClientModel.combiner.in_(combiners))

            items = session.execute(stmt).fetchall()

            result = []
            for item in items:
                result.append({"combiner": item[0], "count": item[1]})

            return result

    def _to_orm_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("client_id", None)
        return item_dict

    def _dto_from_orm_model(self, item: ClientModel) -> ClientDTO:
        orm_dict = from_orm_model(item, ClientModel)
        orm_dict["client_id"] = orm_dict.pop("id")
        return ClientDTO().populate_with(orm_dict)
