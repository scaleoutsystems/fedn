from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import String, func, or_, select
from sqlalchemy.orm import Mapped, mapped_column

from fedn.network.storage.statestore.stores.store import MongoDBStore, MyAbstractBase, SQLStore, Store

from .shared import from_document


class Combiner:
    def __init__(
        self,
        id: str,
        name: str,
        address: str,
        certificate: str,
        config: dict,
        fqdn: str,
        ip: str,
        key: str,
        parent: dict,
        port: int,
        status: str,
        updated_at: str,
    ):
        self.id = id
        self.name = name
        self.address = address
        self.certificate = certificate
        self.config = config
        self.fqdn = fqdn
        self.ip = ip
        self.key = key
        self.parent = parent
        self.port = port
        self.status = status
        self.updated_at = updated_at


class CombinerStore(Store[Combiner]):
    pass


class MongoDBCombinerStore(MongoDBStore[Combiner]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Combiner:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the name (property)
        return: The entity
        """
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            document = self.database[self.collection].find_one({"_id": id_obj})
        else:
            document = self.database[self.collection].find_one({"name": id})

        if document is None:
            return None

        return from_document(document)

    def update(self, id: str, item: Combiner) -> bool:
        raise NotImplementedError("Update not implemented for CombinerStore")

    def add(self, item: Combiner) -> Tuple[bool, Any]:
        return super().add(item)

    def delete(self, id: str) -> bool:
        if ObjectId.is_valid(id):
            kwargs = {"_id": ObjectId(id)}
        else:
            return False

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            return False

        return super().delete(document["_id"])

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Combiner]]:
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
        response = super().list(limit, skip, sort_key or "updated_at", sort_order, **kwargs)

        return response

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)


class CombinerModel(MyAbstractBase):
    __tablename__ = "combiners"

    address: Mapped[str] = mapped_column(String(255))
    fqdn: Mapped[Optional[str]] = mapped_column(String(255))
    ip: Mapped[Optional[str]] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    parent: Mapped[Optional[str]] = mapped_column(String(255))
    port: Mapped[int]
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now())


def from_row(row: CombinerModel) -> Combiner:
    return {
        "id": row.id,
        "committed_at": row.committed_at,
        "address": row.address,
        "ip": row.ip,
        "name": row.name,
        "parent": row.parent,
        "fqdn": row.fqdn,
        "port": row.port,
        "updated_at": row.updated_at,
    }


class SQLCombinerStore(CombinerStore, SQLStore[Combiner]):
    def __init__(self, Session):
        super().__init__(Session)

    def get(self, id: str) -> Combiner:
        with self.Session() as session:
            stmt = select(CombinerModel).where(or_(CombinerModel.id == id, CombinerModel.name == id))
            item = session.scalars(stmt).first()
            if item is None:
                return None
            return from_row(item)

    def update(self, id, item):
        raise NotImplementedError

    def add(self, item):
        with self.Session() as session:
            entity = CombinerModel(
                address=item["address"],
                fqdn=item["fqdn"],
                ip=item["ip"],
                name=item["name"],
                parent=item["parent"],
                port=item["port"],
            )
            session.add(entity)
            session.commit()
            return True, from_row(entity)

    def delete(self, id: str) -> bool:
        with self.Session() as session:
            stmt = select(CombinerModel).where(CombinerModel.id == id)
            item = session.scalars(stmt).first()
            if item is None:
                return False
            session.delete(item)
            return True

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            stmt = select(CombinerModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(CombinerModel, key) == value)

            _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
            _sort_key: str = sort_key or "committed_at"

            if _sort_key in CombinerModel.__table__.columns:
                sort_obj = CombinerModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else CombinerModel.__table__.columns.get(_sort_key).desc()

                stmt = stmt.order_by(sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            elif skip:
                stmt = stmt.offset(skip)

            items = session.scalars(stmt).all()

            result = []
            for i in items:
                result.append(from_row(i))

            count = session.scalar(select(func.count()).select_from(CombinerModel))

            return {"count": count, "result": result}

    def count(self, **kwargs):
        with self.Session() as session:
            stmt = select(func.count()).select_from(CombinerModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(CombinerModel, key) == value)

            count = session.scalar(stmt)

            return count
