import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from werkzeug.utils import secure_filename

from fedn.network.storage.statestore.stores.store import Store

from .shared import EntityNotFound, from_document


class Package:
    def __init__(
        self, id: str, key: str, committed_at: datetime, description: str, file_name: str, helper: str, name: str, storage_file_name: str, active: bool = False
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
            active=active,
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

    def _validate(self, item: Package) -> Tuple[bool, str]:
        if "file_name" not in item or not item["file_name"]:
            return False, "File name is required"

        if not self._allowed_file_extension(item["file_name"]):
            return False, "File extension not allowed"

        if "helper" not in item or not item["helper"]:
            return False, "Helper is required"

        return True, ""

    def _complement(self, item: Package):
        if "id" not in item or item.id is None:
            item["id"] = str(uuid.uuid4())

        if "key" not in item or item.key is None:
            item["key"] = "package_trail"

        if "committed_at" not in item or item.committed_at is None:
            item["committed_at"] = datetime.now()

        extension = item["file_name"].rsplit(".", 1)[1].lower()

        if "storage_file_name" not in item or item.storage_file_name is None:
            storage_file_name = secure_filename(f"{str(uuid.uuid4())}.{extension}")
            item["storage_file_name"] = storage_file_name

    def set_active(self, id: str) -> bool:
        """Set the active entity
        param id: The id of the entity
            type: str
        return: Whether the operation was successful
        """
        kwargs = {"_id": ObjectId(id)} if ObjectId.is_valid(id) else {"id": id}
        kwargs["key"] = "package_trail"

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with id {id} not found")

        committed_at = datetime.now()
        obj_to_insert = {
            "key": "active",
            "id": document["id"],
            "committed_at": committed_at,
            "description": document["description"],
            "file_name": document["file_name"],
            "helper": document["helper"],
            "name": document["name"],
            "storage_file_name": document["storage_file_name"],
        }

        self.database[self.collection].update_one({"key": "active"}, {"$set": obj_to_insert}, upsert=True)

        return True

    def get_active(self, use_typing: bool = False) -> Package:
        """Get the active entity
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entity
        """
        kwargs = {"key": "active"}
        response = self.database[self.collection].find_one(kwargs)

        if response is None:
            raise EntityNotFound("Entity not found")

        return Package.from_dict(response, response) if use_typing else from_document(response)

    def set_active_helper(self, helper: str) -> bool:
        """Set the active helper
        param helper: The helper to set as active
            type: str
        return: Whether the operation was successful
        """
        if not helper or helper == "" or helper not in ["numpyhelper", "binaryhelper", "androidhelper"]:
            raise ValueError()

        try:
            self.database[self.collection].update_one({"key": "active"}, {"$set": {"helper": helper}}, upsert=True)
        except Exception:
            return False

    def _allowed_file_extension(self, filename: str, ALLOWED_EXTENSIONS={"gz", "bz2", "tar", "zip", "tgz"}) -> bool:
        """Check if file extension is allowed.

        :param filename: The filename to check.
        :type filename: str
        :return: True and extension str if file extension is allowed, else False and None.
        :rtype: Tuple (bool, str)
        """
        if "." in filename:
            extension = filename.rsplit(".", 1)[1].lower()
            if extension in ALLOWED_EXTENSIONS:
                return True

        return False

    def update(self, id: str, item: Package) -> bool:
        raise NotImplementedError("Update not implemented for PackageStore")

    def add(self, item: Package) -> Tuple[bool, Any]:
        valid, message = self._validate(item)
        if not valid:
            return False, message

        self._complement(item)

        return super().add(item)

    def delete(self, id: str) -> bool:
        kwargs = {"_id": ObjectId(id)} if ObjectId.is_valid(id) else {"id": id}
        kwargs["key"] = "package_trail"
        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            raise EntityNotFound(f"Entity with (id) {id} not found")

        result = super().delete(document["_id"])

        if not result:
            return False

        kwargs["key"] = "active"

        document_active = self.database[self.collection].find_one(kwargs)

        if document_active is not None:
            return super().delete(document_active["_id"])

        return True

    def delete_active(self):
        kwargs = {"key": "active"}

        document_active = self.database[self.collection].find_one(kwargs)

        if document_active is None:
            raise EntityNotFound("Entity not found")

        return super().delete(document_active["_id"])

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

        return {"count": response["count"], "result": result}

    def count(self, **kwargs) -> int:
        kwargs["key"] = "package_trail"
        return super().count(**kwargs)
