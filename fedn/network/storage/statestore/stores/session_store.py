from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import select

from fedn.network.storage.statestore.stores.dto.session import SessionConfigDTO, SessionDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
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


def validate(item: dict) -> Tuple[bool, str]:
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
        entity = self.mongo_get(id)
        if entity is None:
            return None
        return self._dto_from_document(entity)

    def update(self, item: SessionDTO) -> Tuple[bool, Any]:
        item_dict = item.to_db(exclude_unset=False)
        valid, message = validate(item_dict)
        if not valid:
            return False, message

        success, obj = self.mongo_update(item_dict)
        if success:
            return success, self._dto_from_document(obj)

        return success, obj

    def add(self, item: SessionDTO) -> Tuple[bool, Any]:
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
        session_config_dto = SessionConfigDTO().patch_with(session["session_config"], throw_on_extra_keys=False)
        session["session_config"] = session_config_dto

        return SessionDTO().patch_with(session, throw_on_extra_keys=False)


class SQLSessionStore(SessionStore, SQLStore[SessionModel]):
    def __init__(self, Session):
        super().__init__(Session, SessionModel)

    def get(self, id: str) -> SessionDTO:
        with self.Session() as session:
            entity = self.sql_get(session, id)
            if entity is None:
                return None

            return self._dto_from_orm_model(entity)

    def update(self, item: SessionDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=True)
            item_dict = self._to_orm_dict(item_dict)
            valid, message = validate(item_dict)
            if not valid:
                return False, message

            session_config_dict = item_dict.pop("session_config")

            stmt = select(SessionModel).where(SessionModel.id == item_dict["id"])
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                return False, "Item not found"

            for key, value in item_dict.items():
                setattr(existing_item, key, value)

            exsisting_session_config = existing_item.session_config

            for key, value in session_config_dict.items():
                setattr(exsisting_session_config, key, value)

            session.commit()

            return True, self._dto_from_orm_model(existing_item)

    def add(self, item: SessionDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_db(exclude_unset=False)
            item_dict = self._to_orm_dict(item_dict)

            valid, message = validate(item_dict)
            if not valid:
                return False, message

            session_config_dict = item_dict.pop("session_config")

            parent_entity = SessionModel(**item_dict)
            child_entity = SessionConfigModel(**session_config_dict)
            parent_entity.session_config = child_entity

            session.add(child_entity)
            session.add(parent_entity)

            session.commit()

            return True, self._dto_from_orm_model(parent_entity)

    def delete(self, id: str) -> bool:
        with self.Session() as session:
            stmt = select(SessionModel).where(SessionModel.id == id)
            item = session.scalars(stmt).first()

            if item is None:
                return False

            session_config = item.session_config

            session.delete(item)
            session.delete(session_config)
            session.commit()

            return True

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[SessionDTO]:
        with self.Session() as session:
            entities = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in entities]

    def count(self, **kwargs):
        return self.sql_count(**kwargs)

    def _dto_from_orm_model(self, item: SessionModel) -> SessionDTO:
        session_dict = from_orm_model(item, SessionModel)
        session_config_dict = from_orm_model(item.session_config, SessionConfigModel)

        session_dict["session_config"] = session_config_dict
        session_dict["session_id"] = session_dict.pop("id")

        session_config_dict.pop("id")
        session_config_dict.pop("committed_at")
        session_dict.pop("session_config_id")
        return SessionDTO().populate_with(session_dict)

    def _to_orm_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("session_id", None)
        return item_dict
