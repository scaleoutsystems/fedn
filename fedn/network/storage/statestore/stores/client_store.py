from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select

from fedn.network.storage.statestore.stores.dto import Client
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store
from fedn.network.storage.statestore.stores.sql.shared import ClientModel


class ClientStore(Store[Client]):
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


class MongoDBClientStore(ClientStore, MongoDBStore[Client]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "client_id", Client)

    def get(self, client_id: str) -> Client:
        return super().get(client_id)

    def add(self, item: Client) -> Tuple[bool, Any]:
        return super().add(item)

    def update(self, item: Client) -> Tuple[bool, Any]:
        return super().update(item)

    def delete(self, client_id: str) -> bool:
        return super().delete(client_id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **filter_kwargs) -> List[Client]:
        return super().select(limit, skip, sort_key, sort_order, **filter_kwargs)

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)

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


class SQLClientStore(ClientStore, SQLStore[Client]):
    def __init__(self, Session):
        super().__init__(Session, "client_id", ClientModel, Client)

    def get(self, id: str) -> Client:
        return super().get(id)

    def add(self, item: Client) -> Tuple[bool, Any]:
        return super().add(item)

    def update(self, item: Client) -> Tuple[bool, Any]:
        return super().update(item)

    def delete(self, id) -> bool:
        return super().delete(id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[Client]:
        return super().select(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        return super().count(**kwargs)

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
