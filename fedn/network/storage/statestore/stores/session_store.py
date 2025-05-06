from typing import Dict

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.session import SessionConfigDTO, SessionDTO
from fedn.network.storage.statestore.stores.sql.shared import SessionConfigModel, SessionModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class SessionStore(Store[SessionDTO]):
    pass


class MongoDBSessionStore(SessionStore, MongoDBStore[SessionDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "session_id")

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
        super().__init__(Session, SessionModel, "session_id")

    def update(self, item: SessionDTO) -> SessionDTO:
        return self.sql_update(item)

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
        session_config_dict.pop("updated_at")
        session_dict.pop("session_config_id")

        return SessionDTO().populate_with(session_dict)

    def _to_orm_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("session_id", None)
        return item_dict
