from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import Store

from .shared import EntityNotFound, from_document


class Package:
    def __init__(
            self,
            id: str,
            key: str,
            committed_at: datetime,
            description: str,
            file_name: str,
            helper: str,
            name: str,
            storage_file_name: str,
            active: bool = False
    ):
        self.key = key
        self.committed_at = committed_at
        self.description = description
        self.file_name = file_name
        self.helper = helper
        self.id = id
        self.name = name
        self.storage_file_name = storage_file_name
        self.active = active

    def from_dict(data: dict, active_package: dict) -> "Package":
        active = False
        if active_package:
            if "id" in active_package and "id" in data:
                active = active_package["id"] == data["id"]

        return Package(
            id=data["id"] if "id" in data else None,
            key=data["key"] if "key" in data else None,
            committed_at=data["committed_at"] if "committed_at" in data else None,
            description=data["description"] if "description" in data else None,
            file_name=data["file_name"] if "file_name" in data else None,
            helper=data["helper"] if "helper" in data else None,
            name=data["name"] if "name" in data else None,
            storage_file_name=data["storage_file_name"] if "storage_file_name" in data else None,
            active=active
        )


class PackageStore(Store[Package]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Package:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the model (property)
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
            description: Whether to return the entities as typed objects or as dicts.
            If True, and active property will be set based on the active package.
        return: The entity
        """
        document = self.database[self.collection].find_one({"id": id})

        if document is None:
            raise EntityNotFound(f"Entity with id {id} not found")

        if not use_typing:
            return from_document(document)

        response_active = self.database[self.collection].find_one({"key": "active"})

        return Package.from_dict(document, response_active)

    def get_active(self, use_typing: bool = False) -> Package:
        """Get the active entity
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entity
        """
        response = self.database[self.collection].find_one({"key": "active"})

        if response is None:
            raise EntityNotFound(f"Entity with id {id} not found")

        return Package.from_dict(response, response) if use_typing else from_document(response)

    def update(self, id: str, item: Package) -> bool:
        raise NotImplementedError("Update not implemented for PackageStore")

    def add(self, item: Package)-> Tuple[bool, Any]:
        raise NotImplementedError("Add not implemented for PackageStore")

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for PackageStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Package]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
        param skip: The number of entities to skip
            type: int
        param sort_key: The key to sort by
            type: str
        param sort_order: The order to sort by
            type: pymongo.DESCENDING | pymongo.ASCENDING
        param use_typing: Whether to return the entities as typed objects or as dicts
            type: bool
            description: Whether to return the entities as typed objects or as dicts.
            If True, and active property will be set based on the active package.
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: A dictionary with the count and the result
        """
        kwargs["key"] = "package_trail"

        response = super().list(limit, skip, sort_key or "committed_at", sort_order, use_typing=True, **kwargs)

        response_active = self.database[self.collection].find_one({"key": "active"})

        result = [Package.from_dict(item, response_active) for item in response["result"]]

        return {
            "count": response["count"],
            "result": result
        }

    def count(self, **kwargs) -> int:
        kwargs["key"] = "package_trail"
        return super().count(**kwargs)
