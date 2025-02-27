from typing import Any, Dict, List, Tuple, Type

import pymongo
from pymongo.database import Database
from sqlalchemy import func, select

from fedn.network.storage.statestore.stores.dto.session import SessionConfigDTO, SessionDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store
from fedn.network.storage.statestore.stores.shared import from_document
from fedn.network.storage.statestore.stores.sql.shared import SessionConfigModel, SessionModel, from_orm_model


class SessionStore(Store[SessionDTO]):
    pass


def validate_session_config(session_config: SessionConfigDTO) -> Tuple[bool, str]:
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


def validate(item: SessionDTO) -> Tuple[bool, str]:
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


class MongoDBSessionStore(SessionStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "session_id")
        self.database[self.collection].create_index([("session_id", pymongo.DESCENDING)])

    def get(self, id: str) -> SessionDTO:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the session_id (property)
        return: The entity
        """
        entity = self.mongo_get(id)
        if entity is None:
            return None
        return self._dto_from_document(entity)

    def update(self, item: SessionDTO) -> Tuple[bool, Any]:
        item_dict = item.to_db(exclude_unset=True)
        valid, message = validate(item_dict)
        if not valid:
            return False, message

        success, obj = self.mongo_update(item_dict)
        if success:
            return success, self._dto_from_document(obj)

        return success, obj

    def add(self, item: SessionDTO) -> Tuple[bool, Any]:
        """Add an entity
        param item: The entity to add
            type: SessionDTO
            description: The entity to add
        return: A tuple with a boolean indicating success and the entity
        """
        item_dict = item.to_db(exclude_unset=False)
        success, msg = validate(item_dict)
        if not success:
            return success, msg

        success, obj = self.mongo_add(item_dict)

        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def delete(self, session_id: str) -> bool:
        return self.mongo_delete(session_id)

    def count(self, **kwargs) -> int:
        return self.mongo_count(**kwargs)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **filter_kwargs) -> List[SessionDTO]:
        entites = self.mongo_select(limit, skip, sort_key, sort_order, **filter_kwargs)
        return [self._dto_from_document(entity) for entity in entites]

    def _dto_from_document(self, document: Dict) -> SessionDTO:
        session = from_document(document)
        session_config_dto = SessionConfigDTO().populate_with(session["session_config"])
        session["session_config"] = session_config_dto

        return SessionDTO().populate_with(session)


def from_row(row: dict) -> SessionDTO:
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


class SQLSessionStore(SessionStore, SQLStore[SessionModel]):
    def __init__(self, Session):
        super().__init__(Session, SessionModel)

    def _dto_from_orm_model(self, item: SessionModel) -> SessionDTO:
        orm_dict = from_orm_model(item, SessionModel)
        session_config_dict = from_orm_model(item.session_config, SessionConfigModel)
        session_config = SessionConfigDTO().populate_with(session_config_dict)
        orm_dict["session_config"] = session_config
        orm_dict["session_id"] = orm_dict.pop("id")
        return SessionDTO().populate_with(orm_dict)

    def get(self, id: str) -> SessionDTO:
        with self.Session() as session:
            entity = self.sql_get(session, id)
            if entity is None:
                return None

            return self._dto_from_orm_model(entity)

    def update(self, id: str, item: SessionDTO) -> Tuple[bool, Any]:
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

    def add(self, item: SessionDTO) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message

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
