from typing import Dict, List

import pymongo
from bson import ObjectId
from pymongo.database import Database

from fedn.network.storage.statestore.repositories.repository import Repository

from .shared import from_document


class Combiner:
    def __init__(self, id: str, name: str, address: str, certificate: str | None, config: dict, fqdn: str | None, ip: str, key: str | None, parent: dict, port: int, status: str, updated_at: str):
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

    def from_dict(data: dict) -> 'Combiner':
        return Combiner(
            id=str(data['_id']),
            name=data['name'] if 'name' in data else None,
            address=data['address'] if 'address' in data else None,
            certificate=data['certificate'] if 'certificate' in data else None,
            config=data['config'] if 'config' in data else None,
            fqdn=data['fqdn'] if 'fqdn' in data else None,
            ip=data['ip'] if 'ip' in data else None,
            key=data['key'] if 'key' in data else None,
            parent=data['parent'] if 'parent' in data else None,
            port=data['port'] if 'port' in data else None,
            status=data['status'] if 'status' in data else None,
            updated_at=data['updated_at'] if 'updated_at' in data else None
        )


class CombinerRepository(Repository[Combiner]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Combiner:
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            document = self.database[self.collection].find_one({'_id': id_obj})
        else:
            document = self.database[self.collection].find_one({'name': id})

        if document is None:
            raise KeyError(f"Entity with (id | name) {id} not found")

        return Combiner.from_dict(document) if use_typing else from_document(document)

    def update(self, id: str, item: Combiner) -> bool:
        raise NotImplementedError("Update not implemented for CombinerRepository")

    def add(self, item: Combiner) -> bool:
        raise NotImplementedError("Add not implemented for CombinerRepository")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for CombinerRepository")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Combiner]]:
        response = super().list(limit, skip, sort_key or "updated_at", sort_order, use_typing=use_typing, **kwargs)

        result = [Combiner.from_dict(item) for item in response['result']] if use_typing else response['result']

        return {
            "count": response["count"],
            "result": result
        }

    def count(self, **kwargs) -> int:
        return super().count(**kwargs)
