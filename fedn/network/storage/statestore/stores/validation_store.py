from typing import Any, Dict, List, Optional, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import ForeignKey, String, func, select
from sqlalchemy.orm import Mapped, mapped_column

from fedn.network.storage.statestore.stores.store import MongoDBStore, MyAbstractBase, SQLStore, Store


class Validation:
    def __init__(
        self, id: str, model_id: str, data: str, correlation_id: str, timestamp: str, session_id: str, meta: str, sender: dict = None, receiver: dict = None
    ):
        self.id = id
        self.model_id = model_id
        self.data = data
        self.correlation_id = correlation_id
        self.timestamp = timestamp
        self.session_id = session_id
        self.meta = meta
        self.sender = sender
        self.receiver = receiver


class ValidationStore(Store[Validation]):
    pass


class MongoDBValidationStore(MongoDBStore[Validation]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str) -> Validation:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the validation (property)
        return: The entity
        """
        return super().get(id)

    def update(self, id: str, item: Validation) -> bool:
        raise NotImplementedError("Update not implemented for ValidationStore")

    def add(self, item: Validation) -> Tuple[bool, Any]:
        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for ValidationStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Validation]]:
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
        return: A dictionary with the count and a list of entities
        """
        return super().list(limit, skip, sort_key or "timestamp", sort_order, **kwargs)


class ValidationModel(MyAbstractBase):
    __tablename__ = "validations"

    correlation_id: Mapped[str]
    data: Mapped[Optional[str]]
    model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    receiver_name: Mapped[Optional[str]] = mapped_column(String(255))
    receiver_role: Mapped[Optional[str]] = mapped_column(String(255))
    sender_name: Mapped[Optional[str]] = mapped_column(String(255))
    sender_role: Mapped[Optional[str]] = mapped_column(String(255))
    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    timestamp: Mapped[str] = mapped_column(String(255))


def from_row(row: ValidationModel) -> Validation:
    return {
        "id": row.id,
        "model_id": row.model_id,
        "data": row.data,
        "correlation_id": row.correlation_id,
        "timestamp": row.timestamp,
        "session_id": row.session_id,
        "sender": {"name": row.sender_name, "role": row.sender_role},
        "receiver": {"name": row.receiver_name, "role": row.receiver_role},
    }


class SQLValidationStore(ValidationStore, SQLStore[Validation]):
    def __init__(self, Session):
        super().__init__(Session)

    def get(self, id: str) -> Validation:
        with self.Session() as session:
            stmt = select(ValidationModel).where(ValidationModel.id == id)
            item = session.scalars(stmt).first()

            if item is None:
                return None

            return from_row(item)

    def update(self, id: str, item: Validation) -> bool:
        raise NotImplementedError("Update not implemented for ValidationStore")

    def add(self, item: Validation) -> Tuple[bool, Any]:
        with self.Session() as session:
            sender = item["sender"] if "sender" in item else None
            receiver = item["receiver"] if "receiver" in item else None

            validation = ValidationModel(
                correlation_id=item.get("correlationId") or item.get("correlation_id"),
                data=item.get("data"),
                model_id=item.get("modelId") or item.get("model_id"),
                receiver_name=receiver.get("name") if receiver else None,
                receiver_role=receiver.get("role") if receiver else None,
                sender_name=sender.get("name") if sender else None,
                sender_role=sender.get("role") if sender else None,
                session_id=item.get("sessionId") or item.get("session_id"),
                timestamp=item.get("timestamp"),
            )

            session.add(validation)
            session.commit()

            return True, validation

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for ValidationStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            stmt = select(ValidationModel)

            for key, value in kwargs.items():
                if key == "_id":
                    key = "id"
                elif key == "sender.name":
                    key = "sender_name"
                elif key == "sender.role":
                    key = "sender_role"
                elif key == "receiver.name":
                    key = "receiver_name"
                elif key == "receiver.role":
                    key = "receiver_role"
                elif key == "correlationId":
                    key = "correlation_id"
                elif key == "modelId":
                    key = "model_id"
                elif key == "sessionId":
                    key = "session_id"

                stmt = stmt.where(getattr(ValidationModel, key) == value)

            if sort_key:
                _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
                _sort_key: str = sort_key

                if _sort_key == "_id":
                    _sort_key = "id"
                elif _sort_key == "sender.name":
                    _sort_key = "sender_name"
                elif _sort_key == "sender.role":
                    _sort_key = "sender_role"
                elif _sort_key == "receiver.name":
                    _sort_key = "receiver_name"
                elif _sort_key == "receiver.role":
                    _sort_key = "receiver_role"
                elif _sort_key == "correlationId":
                    _sort_key = "correlation_id"
                elif _sort_key == "modelId":
                    _sort_key = "model_id"
                elif _sort_key == "sessionId":
                    _sort_key = "session_id"

                if _sort_key in ValidationModel.__table__.columns:
                    sort_obj = (
                        ValidationModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else ValidationModel.__table__.columns.get(_sort_key).desc()
                    )

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
            stmt = select(func.count()).select_from(ValidationModel)

            for key, value in kwargs.items():
                if key == "sender.name":
                    key = "sender_name"
                elif key == "sender.role":
                    key = "sender_role"
                elif key == "receiver.name":
                    key = "receiver_name"
                elif key == "receiver.role":
                    key = "receiver_role"
                elif key == "correlationId":
                    key = "correlation_id"
                elif key == "modelId":
                    key = "model_id"
                elif key == "sessionId":
                    key = "session_id"

                stmt = stmt.where(getattr(ValidationModel, key) == value)

            count = session.scalar(stmt)

            return count
