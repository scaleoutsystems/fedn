from typing import Dict, List

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.repositories.repository import Repository

from .shared import EntityNotFound, from_document


class Session:
    def __init__(self, id: str, session_id: str, status: str, session_config: dict = None):
        self.id = id
        self.session_id = session_id
        self.status = status
        self.session_config = session_config

    def from_dict(data: dict) -> 'Session':
        return Session(
            id=str(data['_id']),
            session_id=data['session_id'] if 'session_id' in data else None,
            status=data['status'] if 'status' in data else None,
            session_config=data['session_config'] if 'session_config' in data else None
        )


class SessionRepository(Repository[Session]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Session:
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            document = self.database[self.collection].find_one({'_id': id_obj})
        else:
            document = self.database[self.collection].find_one({'session_id': id})

        if document is None:
            raise EntityNotFound(f"Entity with (id | session_id) {id} not found")

        return Session.from_dict(document) if use_typing else from_document(document)

    def update(self, id: str, item: Session) -> bool:
        raise NotImplementedError("Update not implemented for SessionRepository")

    def add(self, item: Session) -> bool:
        raise NotImplementedError("Add not implemented for SessionRepository")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for SessionRepository")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Session]]:
        response = super().list(limit, skip, sort_key or "session_id", sort_order, use_typing=use_typing, **kwargs)

        result = [Session.from_dict(item) for item in response['result']] if use_typing else response['result']

        return {
            "count": response["count"],
            "result": result
        }
