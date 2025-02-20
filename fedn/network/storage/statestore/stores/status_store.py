from typing import Any, Dict, List, Optional, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import ForeignKey, String, func, select
from sqlalchemy.orm import Mapped, mapped_column

from fedn.network.storage.statestore.stores.store import MongoDBStore, MyAbstractBase, SQLStore, Store


class Status:
    def __init__(
        self, id: str, status: str, timestamp: str, log_level: str, data: str, correlation_id: str, type: str, extra: str, session_id: str, sender: dict = None
    ):
        self.id = id
        self.status = status
        self.timestamp = timestamp
        self.log_level = log_level
        self.data = data
        self.correlation_id = correlation_id
        self.type = type
        self.extra = extra
        self.session_id = session_id
        self.sender = sender


class StatusStore(Store[Status]):
    pass


class MongoDBStatusStore(StatusStore, MongoDBStore[Status]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Status:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the status (property)
        return: The entity
        """
        return super().get(id)

    def update(self, id: str, item: Status) -> bool:
        raise NotImplementedError("Update not implemented for StatusStore")

    def add(self, item: Status) -> Tuple[bool, Any]:
        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for StatusStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Status]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
            description: The maximum number of entities to return
        param skip: The number of entities to skip
            type: int
            description: The number of entities to skip
        param sort_key: The key to sort by
            type: str
            description: The key to sort by
        param sort_order: The order to sort by
            type: pymongo.DESCENDING
            description: The order to sort by
        """
        return super().list(limit, skip, sort_key or "timestamp", sort_order, **kwargs)


class StatusModel(MyAbstractBase):
    __tablename__ = "statuses"

    log_level: Mapped[str] = mapped_column(String(255))
    sender_name: Mapped[Optional[str]] = mapped_column(String(255))
    sender_role: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(255))
    data: Mapped[Optional[str]]
    correlation_id: Mapped[Optional[str]]
    extra: Mapped[Optional[str]]
    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))


def from_row(row: StatusModel) -> Status:
    return {
        "id": row.id,
        "log_level": row.log_level,
        "sender": {"name": row.sender_name, "role": row.sender_role},
        "status": row.status,
        "timestamp": row.timestamp,
        "type": row.type,
        "data": row.data,
        "correlation_id": row.correlation_id,
        "extra": row.extra,
        "session_id": row.session_id,
    }


class SQLStatusStore(StatusStore, SQLStore[Status]):
    def __init__(self, Session):
        super().__init__(Session)

    def get(self, id: str) -> Status:
        with self.Session() as session:
            stmt = select(StatusModel).where(StatusModel.id == id)
            item = session.scalars(stmt).first()

            if item is None:
                return None

            return from_row(item)

    def update(self, id, item):
        raise NotImplementedError

    def add(self, item: Status) -> Tuple[bool, Any]:
        with self.Session() as session:
            sender = item["sender"] if "sender" in item else None

            status = StatusModel(
                log_level=item.get("log_level") or item.get("logLevel"),
                sender_name=sender.get("name") if sender else None,
                sender_role=sender.get("role") if sender else None,
                status=item.get("status"),
                timestamp=item.get("timestamp"),
                type=item.get("type"),
                data=item.get("data"),
                correlation_id=item.get("correlation_id"),
                extra=item.get("extra"),
                session_id=item.get("session_id") or item.get("sessionId"),
            )
            session.add(status)
            session.commit()
            return True, status

    def delete(self, id: str) -> bool:
        raise NotImplementedError

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            stmt = select(StatusModel)

            for key, value in kwargs.items():
                if key == "_id":
                    key = "id"
                elif key == "logLevel":
                    key = "log_level"
                elif key == "sender.name":
                    key = "sender_name"
                elif key == "sender.role":
                    key = "sender_role"
                elif key == "sessionId":
                    key = "session_id"

                stmt = stmt.where(getattr(StatusModel, key) == value)

            if sort_key:
                _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
                _sort_key: str = sort_key

                if _sort_key == "_id":
                    _sort_key = "id"
                elif _sort_key == "logLevel":
                    _sort_key = "log_level"
                elif _sort_key == "sender.name":
                    _sort_key = "sender_name"
                elif _sort_key == "sender.role":
                    _sort_key = "sender_role"
                elif _sort_key == "sessionId":
                    _sort_key = "session_id"

                if _sort_key in StatusModel.__table__.columns:
                    sort_obj = StatusModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else StatusModel.__table__.columns.get(_sort_key).desc()

                    stmt = stmt.order_by(sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            elif skip:
                stmt = stmt.offset(skip)

            items = session.execute(stmt)

            result = []

            for item in items:
                (r,) = item

                result.append(from_row(r))

            return {"count": len(result), "result": result}

    def count(self, **kwargs):
        with self.Session() as session:
            stmt = select(func.count()).select_from(StatusModel)

            for key, value in kwargs.items():
                if key == "_id":
                    key = "id"
                elif key == "logLevel":
                    key = "log_level"
                elif key == "sender.name":
                    key = "sender_name"
                elif key == "sender.role":
                    key = "sender_role"
                elif key == "sessionId":
                    key = "session_id"

                stmt = stmt.where(getattr(StatusModel, key) == value)

            count = session.scalar(stmt)

            return count
