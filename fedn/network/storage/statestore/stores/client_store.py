from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select

from fedn.network.storage.statestore.stores.dto import Client
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import ClientModel, from_sqlalchemy_model


class ClientStore(Store[Client]):
    """Client store interface."""

    @abstractmethod
    def connected_client_count(self, combiners: List[str]) -> List[Client]:
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

    def get(self, client_id: str) -> Client:
        entity = MongoDBStore.get(self, client_id)
        if entity is None:
            return None
        return Client(**from_document(entity))

    def add(self, item: Client) -> Tuple[bool, Any]:
        item_dict = item.to_dict(exclude_unset=False)
        success, obj = MongoDBStore.add(self, item_dict)
        if success:
            return success, Client(**from_document(obj))
        return success, obj

    def update(self, item: Client) -> Tuple[bool, Any]:
        item_dict = item.to_dict()
        success, obj = MongoDBStore.update(self, item_dict)
        if success:
            return success, Client(**from_document(obj))
        return success, obj

    def delete(self, client_id: str) -> bool:
        return MongoDBStore.delete(self, client_id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **filter_kwargs) -> List[Client]:
        entites = MongoDBStore.select(self, limit, skip, sort_key, sort_order, **filter_kwargs)
        return [Client(**from_document(entity)) for entity in entites]

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


class SQLClientStore(ClientStore, SQLStore[ClientModel]):
    def __init__(self, Session):
        super().__init__(Session, "client_id", ClientModel)

    def get(self, id: str) -> Client:
        with self.Session() as session:
            entity = SQLStore.get(self, session, id)
            if entity is None:
                return None
            return Client(**from_sqlalchemy_model(entity, ClientModel))

    def add(self, item: Client) -> Tuple[bool, Any]:
        with self.Session() as session:
            entity = ClientModel(**item.to_dict(exclude_unset=False))
            success, obj = SQLStore.add(self, session, entity)
            if success:
                return success, Client(**from_sqlalchemy_model(obj, ClientModel))
            return success, obj

    def update(self, item: Client) -> Tuple[bool, Any]:
        with self.Session() as session:
            success, obj = SQLStore.update(self, session, item.to_dict())
            if success:
                return success, Client(**from_sqlalchemy_model(obj, ClientModel))
            return success, obj

    def delete(self, id) -> bool:
        with self.Session() as session:
            return SQLStore.delete(self, session, id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[Client]:
        with self.Session() as session:
            entities = SQLStore.select(self, session, limit, skip, sort_key, sort_order, **kwargs)
            return [Client(**from_sqlalchemy_model(item, ClientModel)) for item in entities]

    def count(self, **kwargs):
        with self.Session() as session:
            return SQLStore.count(self, session, **kwargs)

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
