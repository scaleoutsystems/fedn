import datetime
import uuid
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import func, select

from fedn.network.storage.statestore.stores.shared import from_document
from fedn.network.storage.statestore.stores.sql.shared import SessionConfigModel, SessionModel
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store


class SessionConfig:
    def __init__(
        self,
        aggregator: str,
        round_timeout: int,
        buffer_size: int,
        model_id: str,
        delete_models_storage: bool,
        clients_required: int,
        validate: bool,
        helper_type: str,
    ):
        self.aggregator = aggregator
        self.round_timeout = round_timeout
        self.buffer_size = buffer_size
        self.model_id = model_id
        self.delete_models_storage = delete_models_storage
        self.clients_required = clients_required
        self.validate = validate
        self.helper_type = helper_type


class Session:
    def __init__(self, id: str, session_id: str, status: str, committed_at: datetime, name: str = None, session_config: SessionConfig = None):
        self.id = id
        self.session_id = session_id
        self.status = status
        self.committed_at = committed_at
        self.session_config = session_config
        self.name = name


class SessionStore(Store[Session]):
    pass


def validate_session_config(session_config: SessionConfig) -> Tuple[bool, str]:
    if "aggregator" not in session_config:
        return False, "session_config.aggregator is required"

    if "round_timeout" not in session_config:
        return False, "session_config.round_timeout is required"

    if not isinstance(session_config["round_timeout"], (int, float)):
        return False, "session_config.round_timeout must be an integer"

    if "buffer_size" not in session_config:
        return False, "session_config.buffer_size is required"

    if not isinstance(session_config["buffer_size"], int):
        return False, "session_config.buffer_size must be an integer"

    if "model_id" not in session_config or session_config["model_id"] == "":
        return False, "session_config.model_id is required"

    if not isinstance(session_config["model_id"], str):
        return False, "session_config.model_id must be a string"

    if "delete_models_storage" not in session_config:
        return False, "session_config.delete_models_storage is required"

    if not isinstance(session_config["delete_models_storage"], bool):
        return False, "session_config.delete_models_storage must be a boolean"

    if "clients_required" not in session_config:
        return False, "session_config.clients_required is required"

    if not isinstance(session_config["clients_required"], int):
        return False, "session_config.clients_required must be an integer"

    if "validate" not in session_config:
        return False, "session_config.validate is required"

    if not isinstance(session_config["validate"], bool):
        return False, "session_config.validate must be a boolean"

    if "helper_type" not in session_config or session_config["helper_type"] == "":
        return False, "session_config.helper_type is required"

    if not isinstance(session_config["helper_type"], str):
        return False, "session_config.helper_type must be a string"

    return True, ""


def validate(item: Session) -> Tuple[bool, str]:
    if "session_config" not in item or item["session_config"] is None:
        return False, "session_config is required"

    session_config = None

    if isinstance(item["session_config"], dict):
        session_config = item["session_config"]
    elif isinstance(item["session_config"], list):
        session_config = item["session_config"][0]
    else:
        return False, "session_config must be a dict"

    return validate_session_config(session_config)


def complement(item: Session):
    item["status"] = "Created"
    item["committed_at"] = datetime.datetime.now()

    if "session_id" not in item or item["session_id"] == "" or not isinstance(item["session_id"], str):
        item["session_id"] = str(uuid.uuid4())


class MongoDBSessionStore(MongoDBStore[Session]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("session_id", pymongo.DESCENDING)])

    def get(self, id: str) -> Session:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the session_id (property)
        return: The entity
        """
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            document = self.database[self.collection].find_one({"_id": id_obj})
        else:
            document = self.database[self.collection].find_one({"session_id": id})

        if document is None:
            return None

        return from_document(document)

    def update(self, id: str, item: Session) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message

        return super().update(id, item)

    def add(self, item: Session) -> Tuple[bool, Any]:
        """Add an entity
        param item: The entity to add
            type: Session
            description: The entity to add
        return: A tuple with a boolean indicating success and the entity
        """
        valid, message = validate(item)
        if not valid:
            return False, message

        complement(item)

        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for SessionStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Session]]:
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
        param kwargs: Additional query parameters
            type: dict
            description: Additional query parameters
        return: The entities
        """
        return super().list(limit, skip, sort_key or "session_id", sort_order, **kwargs)


def from_row(row: dict) -> Session:
    return {
        "id": row["id"],
        "name": row["name"],
        "session_id": row["id"],
        "status": row["status"],
        "committed_at": row["committed_at"],
        "session_config": {
            "aggregator": row["aggregator"],
            "round_timeout": row["round_timeout"],
            "buffer_size": row["buffer_size"],
            "model_id": row["model_id"],
            "delete_models_storage": row["delete_models_storage"],
            "clients_required": row["clients_required"],
            "validate": row["validate"],
            "helper_type": row["helper_type"],
        },
    }


class SQLSessionStore(SessionStore, SQLStore[Session]):
    def __init__(self, Session):
        super().__init__(Session)

    def get(self, id: str) -> Session:
        with self.Session() as session:
            stmt = select(SessionModel, SessionConfigModel).join(SessionModel.session_config).where(SessionModel.id == id)
            item = session.execute(stmt).first()
            if item is None:
                return None
            s, c = item
            combined_dict = {
                "id": s.id,
                "name": s.name,
                "session_id": s.id,
                "status": s.status,
                "committed_at": s.committed_at,
                "aggregator": c.aggregator,
                "round_timeout": c.round_timeout,
                "buffer_size": c.buffer_size,
                "model_id": c.model_id,
                "delete_models_storage": c.delete_models_storage,
                "clients_required": c.clients_required,
                "validate": c.validate,
                "helper_type": c.helper_type,
            }
            return from_row(combined_dict)

    def update(self, id: str, item: Session) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message
        with self.Session() as session:
            stmt = select(SessionModel, SessionConfigModel).join(SessionModel.session_config).where(SessionModel.id == id)
            existing_item = session.execute(stmt).first()
            if existing_item is None:
                return False, f"Entity not found {id}"
            s, c = existing_item

            s.name = item["name"] if "name" in item else None
            s.status = item["status"]
            s.committed_at = item["committed_at"]

            session_config = item["session_config"]

            c.aggregator = session_config["aggregator"]
            c.round_timeout = session_config["round_timeout"]
            c.buffer_size = session_config["buffer_size"]
            c.model_id = session_config["model_id"]
            c.delete_models_storage = session_config["delete_models_storage"]
            c.clients_required = session_config["clients_required"]
            c.validate = session_config["validate"]
            c.helper_type = session_config["helper_type"]

            session.commit()

            combined_dict = {
                "id": s.id,
                "name": s.name,
                "session_id": s.id,
                "status": s.status,
                "committed_at": s.committed_at,
                "aggregator": c.aggregator,
                "round_timeout": c.round_timeout,
                "buffer_size": c.buffer_size,
                "model_id": c.model_id,
                "delete_models_storage": c.delete_models_storage,
                "clients_required": c.clients_required,
                "validate": c.validate,
                "helper_type": c.helper_type,
            }

            return True, from_row(combined_dict)

    def add(self, item: Session) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message

        complement(item)

        with self.Session() as session:
            parent_item = SessionModel(
                id=item["session_id"], status=item["status"], name=item["name"] if "name" in item else None, committed_at=item["committed_at"] or None
            )

            session_config = item["session_config"]

            child_item = SessionConfigModel(
                aggregator=session_config["aggregator"],
                round_timeout=session_config["round_timeout"],
                buffer_size=session_config["buffer_size"],
                model_id=session_config["model_id"],
                delete_models_storage=session_config["delete_models_storage"],
                clients_required=session_config["clients_required"],
                validate=session_config["validate"],
                helper_type=session_config["helper_type"],
            )
            child_item.session = parent_item

            session.add(child_item)
            session.add(parent_item)
            session.commit()

            combined_dict = {
                "id": parent_item.id,
                "name": parent_item.name,
                "session_id": parent_item.id,
                "status": parent_item.status,
                "committed_at": parent_item.committed_at,
                "aggregator": child_item.aggregator,
                "round_timeout": child_item.round_timeout,
                "buffer_size": child_item.buffer_size,
                "model_id": child_item.model_id,
                "delete_models_storage": child_item.delete_models_storage,
                "clients_required": child_item.clients_required,
                "validate": child_item.validate,
                "helper_type": child_item.helper_type,
            }

            return True, from_row(combined_dict)

    def delete(self, id: str) -> bool:
        raise NotImplementedError

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            stmt = select(SessionModel, SessionConfigModel).join(SessionModel.session_config)
            for key, value in kwargs.items():
                if "session_config" in key:
                    key = key.replace("session_config.", "")
                    stmt = stmt.where(getattr(SessionConfigModel, key) == value)
                else:
                    stmt = stmt.where(getattr(SessionModel, key) == value)

            if sort_key:
                _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
                _sort_key: str = sort_key or "committed_at"

                if _sort_key in SessionModel.__table__.columns:
                    sort_obj = SessionModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else SessionModel.__table__.columns.get(_sort_key).desc()

                    stmt = stmt.order_by(sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            elif skip:
                stmt = stmt.offset(skip)

            items = session.execute(stmt)

            result = []

            for item in items:
                s, c = item
                combined_dict = {
                    "id": s.id,
                    "session_id": s.id,
                    "name": s.name,
                    "status": s.status,
                    "committed_at": s.committed_at,
                    "aggregator": c.aggregator,
                    "round_timeout": c.round_timeout,
                    "buffer_size": c.buffer_size,
                    "model_id": c.model_id,
                    "delete_models_storage": c.delete_models_storage,
                    "clients_required": c.clients_required,
                    "validate": c.validate,
                    "helper_type": c.helper_type,
                }
                result.append(from_row(combined_dict))

            return {"count": len(result), "result": result}

    def count(self, **kwargs):
        with self.Session() as session:
            stmt = select(func.count()).select_from(SessionModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(SessionModel, key) == value)

            count = session.scalar(stmt)

            return count
