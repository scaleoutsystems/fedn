from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import select

from fedn.network.storage.statestore.stores.dto.session import SessionConfigDTO, SessionDTO
from fedn.network.storage.statestore.stores.sql.shared import SessionConfigModel, SessionModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


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


class MongoDBSessionStore(SessionStore, MongoDBStore[SessionDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "session_id")
        self.database[self.collection].create_index([("session_id", pymongo.DESCENDING)])

    def update(self, item: SessionDTO) -> SessionDTO:
        return self.mongo_update(item)

    def _document_from_dto(self, item: SessionDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> SessionDTO:
        session = from_document(document)
        session_config_dto = SessionConfigDTO().patch_with(session["session_config"], throw_on_extra_keys=False)
        session["session_config"] = session_config_dto

        return SessionDTO().patch_with(session, throw_on_extra_keys=False)


class SQLSessionStore(SessionStore, SQLStore[SessionDTO, SessionModel]):
    def __init__(self, Session):
        super().__init__(Session, SessionModel)

    def update(self, item: SessionDTO) -> SessionDTO:
        return self.sql_update(item)

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

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **kwargs) -> List[SessionDTO]:
        with self.Session() as session:
            entities = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in entities]

    def count(self, **kwargs):
        return self.sql_count(**kwargs)

    def _update_orm_model_from_dto(self, entity: SessionModel, item: SessionDTO) -> SessionModel:
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("session_id", None)

        session_config_dict = item_dict.pop("session_config")
        if entity.session_config is None:
            entity.session_config = SessionConfigModel(**session_config_dict)
        else:
            for key, value in session_config_dict.items():
                setattr(entity.session_config, key, value)

        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

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
