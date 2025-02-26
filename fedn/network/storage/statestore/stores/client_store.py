from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import String, func, or_, select
from sqlalchemy.orm import Mapped, mapped_column

from fedn.network.storage.statestore.stores.store import MongoDBStore, MyAbstractBase, SQLStore, Store

from .shared import from_document


class Client:
    def __init__(self, id: str, name: str, combiner: str, combiner_preferred: str, ip: str, status: str, updated_at: str, last_seen: datetime):
        self.id = id
        self.name = name
        self.combiner = combiner
        self.combiner_preferred = combiner_preferred
        self.ip = ip
        self.status = status
        self.updated_at = updated_at
        self.last_seen = last_seen


class ClientStore(Store[Client]):
    @abstractmethod
    def upsert(self, item: Client) -> Tuple[bool, Any]:
        pass

    @abstractmethod
    def connected_client_count(self, combiners: List[str]) -> List[Client]:
        pass


class MongoDBClientStore(ClientStore, MongoDBStore[Client]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("client_id", pymongo.DESCENDING)])

    def get(self, id: str) -> Client:
        """Get an entity by id
        param id: The id of the entity
            type: str
        return: The entity
        """
        if ObjectId.is_valid(id):
            response = super().get(id)
        else:
            obj = self._get_client_by_client_id(id)
            response = from_document(obj)
        return response

    def _get_client_by_client_id(self, client_id: str) -> Dict:
        document = self.database[self.collection].find_one({"client_id": client_id})
        if document is None:
            return None
        return document

    def update(self, id: str, item: Client) -> Tuple[bool, Any]:
        try:
            existing_client = self.get(id)

            return super().update(existing_client["id"], item)
        except Exception as e:
            return False, str(e)

    def add(self, item: Client) -> Tuple[bool, Any]:
        return super().add(item)

    def upsert(self, item: Client) -> Tuple[bool, Any]:
        try:
            result = self.database[self.collection].update_one(
                {"client_id": item["client_id"]}, {"$set": {k: v for k, v in item.items() if v is not None}}, upsert=True
            )
            id = result.upserted_id
            document = self.database[self.collection].find_one({"_id": id})
            return True, from_document(document)
        except Exception as e:
            return False, str(e)

    def delete(self, id: str) -> bool:
        kwargs = {"_id": ObjectId(id)} if ObjectId.is_valid(id) else {"client_id": id}

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            return False

        return super().delete(document["_id"])

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Client]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
        param skip: The number of entities to skip
            type: int
        param sort_key: The key to sort by
            type: str
        param sort_order: The order to sort by
            type: pymongo.DESCENDING | pymongo.ASCENDING
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: A dictionary with the count and the result
        """
        return super().list(limit, skip, sort_key or "last_seen", sort_order, **kwargs)

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)

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
            result = {}

        return result


class ClientModel(MyAbstractBase):
    __tablename__ = "clients"

    client_id: Mapped[str] = mapped_column(String(255), unique=True)
    combiner: Mapped[str] = mapped_column(String(255))
    ip: Mapped[Optional[str]] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    package: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
    last_seen: Mapped[datetime] = mapped_column(default=datetime.now())


def from_row(row: ClientModel) -> Client:
    return {
        "id": row.id,
        "client_id": row.client_id,
        "combiner": row.combiner,
        "ip": row.ip,
        "name": row.name,
        "package": row.package,
        "status": row.status,
        "last_seen": row.last_seen,
    }


class SQLClientStore(ClientStore, SQLStore[Client]):
    def __init__(self, Session):
        super().__init__(Session)

    def get(self, id: str) -> Client:
        with self.Session() as session:
            stmt = select(ClientModel).where(or_(ClientModel.id == id, ClientModel.client_id == id))
            item = session.scalars(stmt).first()

            if item is None:
                return None

            return from_row(item)

    def update(self, id: str, item: Client) -> Tuple[bool, Any]:
        with self.Session() as session:
            stmt = select(ClientModel).where(or_(ClientModel.id == id, ClientModel.client_id == id))
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                return False, "Item not found"

            existing_item.combiner = item.get("combiner")
            existing_item.ip = item.get("ip")
            existing_item.name = item.get("name")
            existing_item.package = item.get("package")
            existing_item.status = item.get("status")
            existing_item.last_seen = item.get("last_seen")

            session.commit()

            return True, from_row(existing_item)

    def add(self, item: Client) -> Tuple[bool, Any]:
        with self.Session() as session:
            entity = ClientModel(
                client_id=item.get("client_id"),
                combiner=item.get("combiner"),
                ip=item.get("ip"),
                name=item.get("name"),
                package=item.get("package"),
                status=item.get("status"),
                last_seen=item.get("last_seen"),
            )

            session.add(entity)
            session.commit()

            return True, from_row(entity)

    def delete(self, id):
        raise NotImplementedError

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            stmt = select(ClientModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(ClientModel, key) == value)

            _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
            _sort_key: str = sort_key or "committed_at"

            if _sort_key in ClientModel.__table__.columns:
                sort_obj = ClientModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else ClientModel.__table__.columns.get(_sort_key).desc()

                stmt = stmt.order_by(sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            elif skip:
                stmt = stmt.offset(skip)

            items = session.scalars(stmt).all()

            result = []
            for i in items:
                result.append(from_row(i))

            count = session.scalar(select(func.count()).select_from(ClientModel))

            return {"count": count, "result": result}

    def count(self, **kwargs):
        with self.Session() as session:
            stmt = select(func.count()).select_from(ClientModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(ClientModel, key) == value)

            count = session.scalar(stmt)

            return count

    def upsert(self, item: Client) -> Tuple[bool, Any]:
        with self.Session() as session:
            id = item.get("id")
            client_id = item.get("client_id")

            stmt = select(ClientModel).where(or_(ClientModel.id == id, ClientModel.client_id == client_id))
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                entity = ClientModel(
                    client_id=item.get("client_id"),
                    combiner=item.get("combiner"),
                    ip=item.get("ip"),
                    name=item.get("name"),
                    package=item.get("package"),
                    status=item.get("status"),
                    last_seen=item.get("last_seen"),
                )

                session.add(entity)
                session.commit()

                return True, from_row(entity)

            existing_item.combiner = item.get("combiner")
            existing_item.ip = item.get("ip")
            existing_item.name = item.get("name")
            existing_item.package = item.get("package")
            existing_item.status = item.get("status")
            existing_item.last_seen = item.get("last_seen")

            session.commit()

            return True, from_row(existing_item)

    def connected_client_count(self, combiners):
        with self.Session() as session:
            stmt = select(ClientModel.combiner, func.count(ClientModel.combiner)).group_by(ClientModel.combiner)
            if combiners:
                stmt = stmt.where(ClientModel.combiner.in_(combiners))

            items = session.execute(stmt).fetchall()

            result = []
            for i in items:
                result.append({"combiner": i[0], "count": i[1]})

            return result
