from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select

from fedn.network.storage.statestore.stores.dto import ClientDTO
from fedn.network.storage.statestore.stores.sql.shared import ClientModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


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


class MongoDBClientStore(ClientStore, MongoDBStore[ClientDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "client_id")

    def update(self, item: ClientDTO) -> ClientDTO:
        return self.mongo_update(item)

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
        return ClientDTO().patch_with(from_document(document), throw_on_extra_keys=False)

    def _document_from_dto(self, item):
        return item.to_db(exclude_unset=False)


class SQLClientStore(ClientStore, SQLStore[ClientDTO, ClientModel]):
    def __init__(self, Session):
        super().__init__(Session, ClientModel)

    def update(self, item: ClientDTO) -> ClientDTO:
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=True)
            item_dict["id"] = item_dict.pop("client_id", None)
            success, obj = self.sql_update(session, item_dict)
            return self._dto_from_orm_model(obj)

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[ClientDTO]:
        with self.Session() as session:
            entities = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in entities]

    def count(self, **kwargs):
        return self.sql_count(**kwargs)

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

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

    def _update_orm_model_from_dto(self, entity: ClientModel, item: ClientDTO) -> ClientModel:
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("client_id", None)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: ClientModel) -> ClientDTO:
        orm_dict = from_orm_model(item, ClientModel)
        orm_dict["client_id"] = orm_dict.pop("id")
        return ClientDTO().populate_with(orm_dict)
