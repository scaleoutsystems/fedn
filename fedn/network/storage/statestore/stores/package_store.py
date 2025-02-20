import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import String, func, select
from sqlalchemy.orm import Mapped, mapped_column
from werkzeug.utils import secure_filename

from fedn.network.storage.statestore.stores.store import MongoDBStore, MyAbstractBase, SQLStore, Store


def from_document(data: dict, active_package: dict):
    active = False
    if active_package:
        if "id" in active_package and "id" in data:
            active = active_package["id"] == data["id"]

    return {
        "id": data["id"] if "id" in data else None,
        "key": data["key"] if "key" in data else None,
        "committed_at": data["committed_at"] if "committed_at" in data else None,
        "description": data["description"] if "description" in data else None,
        "file_name": data["file_name"] if "file_name" in data else None,
        "helper": data["helper"] if "helper" in data else None,
        "name": data["name"] if "name" in data else None,
        "storage_file_name": data["storage_file_name"] if "storage_file_name" in data else None,
        "active": active,
    }


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
        active: bool = False,
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


class PackageStore(Store[Package]):
    @abstractmethod
    def set_active(self, id: str) -> bool:
        pass

    @abstractmethod
    def get_active(self) -> Package:
        pass

    @abstractmethod
    def set_active_helper(self, helper: str) -> bool:
        pass

    @abstractmethod
    def delete_active(self):
        pass


def allowed_file_extension(filename: str, ALLOWED_EXTENSIONS={"gz", "bz2", "tar", "zip", "tgz"}) -> bool:
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


def validate(item: Package) -> Tuple[bool, str]:
    if "file_name" not in item or not item["file_name"]:
        return False, "File name is required"

    if not allowed_file_extension(item["file_name"]):
        return False, "File extension not allowed"

    if "helper" not in item or not item["helper"]:
        return False, "Helper is required"

    return True, ""


class MongoDBPackageStore(PackageStore, MongoDBStore[Package]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("id", pymongo.DESCENDING)])

    def get(self, id: str) -> Package:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the docuemnt _id
        return: The entity
        """
        document = self.database[self.collection].find_one({"id": id})

        if document is None:
            return None

        response_active = self.database[self.collection].find_one({"key": "active"})

        return from_document(document, response_active)

    def _complement(self, item: Package):
        if "id" not in item or item["id"] is None:
            item["id"] = str(uuid.uuid4())

        if "key" not in item or item["key"] is None:
            item["key"] = "package_trail"

        if "committed_at" not in item or item["committed_at"] is None:
            item["committed_at"] = datetime.now()

        extension = item["file_name"].rsplit(".", 1)[1].lower()

        if "storage_file_name" not in item or item["storage_file_name"] is None:
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
            return False

        committed_at = datetime.now()
        obj_to_insert = {
            "key": "active",
            "id": document["id"] if "id" in document else "",
            "committed_at": committed_at,
            "description": document["description"] if "description" in document else "",
            "file_name": document["file_name"] if "file_name" in document else "",
            "helper": document["helper"] if "helper" in document else "",
            "name": document["name"] if "name" in document else "",
            "storage_file_name": document["storage_file_name"],
        }

        self.database[self.collection].update_one({"key": "active"}, {"$set": obj_to_insert}, upsert=True)

        return True

    def get_active(self) -> Package:
        """Get the active entity
        return: The entity
        """
        kwargs = {"key": "active"}
        response = self.database[self.collection].find_one(kwargs)
        if response is None:
            return None

        active_package = {"id": response["id"]} if "id" in response else {}

        return from_document(response, active_package=active_package)

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

    def update(self, id: str, item: Package) -> bool:
        raise NotImplementedError("Update not implemented for PackageStore")

    def add(self, item: Package) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message

        self._complement(item)

        return super().add(item)

    def delete(self, id: str) -> bool:
        kwargs = {"_id": ObjectId(id)} if ObjectId.is_valid(id) else {"id": id}
        kwargs["key"] = "package_trail"
        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            return False

        result = super().delete(document["_id"])

        if not result:
            return False

        kwargs["key"] = "active"

        document_active = self.database[self.collection].find_one(kwargs)

        if document_active is not None:
            return super().delete(document_active["_id"])

        return True

    def delete_active(self) -> bool:
        kwargs = {"key": "active"}

        document_active = self.database[self.collection].find_one(kwargs)

        if document_active is None:
            return False

        return super().delete(document_active["_id"])

    def list(
        self,
        limit: int,
        skip: int,
        sort_key: str,
        sort_order=pymongo.DESCENDING,
        **kwargs,
    ) -> Dict[int, List[Package]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
        param skip: The number of entities to skip
            type: int
        param sort_key: The key to sort by
            type: str
        param sort_order: The order to sort by
            type: pymongo.DESCENDING | pymongo.ASCENDING
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: A dictionary with the count and the result
        """
        kwargs["key"] = "package_trail"

        response = self.database[self.collection].find(kwargs).sort(sort_key or "committed_at", sort_order).skip(skip or 0).limit(limit or 0)

        count = self.database[self.collection].count_documents(kwargs)

        response_active = self.database[self.collection].find_one({"key": "active"})

        result = [from_document(item, response_active) for item in response]

        return {"count": count, "result": result}

    def count(self, **kwargs) -> int:
        kwargs["key"] = "package_trail"
        return super().count(**kwargs)


class PackageModel(MyAbstractBase):
    __tablename__ = "packages"

    active: Mapped[bool] = mapped_column(default=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    file_name: Mapped[str] = mapped_column(String(255))
    helper: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    storage_file_name: Mapped[str] = mapped_column(String(255))


def from_row(row: PackageModel) -> Package:
    return {
        "id": row.id,
        "committed_at": row.committed_at,
        "description": row.description,
        "file_name": row.file_name,
        "helper": row.helper,
        "name": row.name,
        "storage_file_name": row.storage_file_name,
        "active": row.active,
    }


class SQLPackageStore(PackageStore, SQLStore[Package]):
    def __init__(self, Session):
        super().__init__(Session)

    def _complement(self, item: Package):
        # TODO: Not complemented the same way as in MongoDBStore
        if "committed_at" not in item or item["committed_at"] is None:
            item["committed_at"] = datetime.now()

        extension = item["file_name"].rsplit(".", 1)[1].lower()

        if "storage_file_name" not in item or item["storage_file_name"] is None:
            storage_file_name = secure_filename(f"{str(uuid.uuid4())}.{extension}")
            item["storage_file_name"] = storage_file_name

    def add(self, item: Package) -> Tuple[bool, Any]:
        valid, message = validate(item)
        if not valid:
            return False, message

        self._complement(item)
        with self.Session() as session:
            item = PackageModel(
                committed_at=item["committed_at"],
                description=item["description"] if "description" in item else "",
                file_name=item["file_name"],
                helper=item["helper"],
                name=item["name"] if "name" in item else "",
                storage_file_name=item["storage_file_name"],
            )
            session.add(item)
            session.commit()
            return True, from_row(item)

    def get(self, id: str) -> Package:
        with self.Session() as session:
            stmt = select(PackageModel).where(PackageModel.id == id)
            item = session.scalars(stmt).first()
            if item is None:
                return None
            return from_row(item)

    def update(self, id: str, item: Package) -> bool:
        raise NotImplementedError

    def delete(self, id: str) -> bool:
        with self.Session() as session:
            stmt = select(PackageModel).where(PackageModel.id == id)
            item = session.scalars(stmt).first()
            if item is None:
                return False
            session.delete(item)
            session.commit()
            return True

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            stmt = select(PackageModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(PackageModel, key) == value)

            _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
            _sort_key: str = sort_key or "committed_at"

            if _sort_key in PackageModel.__table__.columns:
                sort_obj = PackageModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else PackageModel.__table__.columns.get(_sort_key).desc()

                stmt = stmt.order_by(sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            elif skip:
                stmt = stmt.offset(skip)

            items = session.scalars(stmt).all()

            result = []
            for i in items:
                result.append(from_row(i))

            count = session.scalar(select(func.count()).select_from(PackageModel))

            return {"count": count, "result": result}

    def count(self, **kwargs):
        with self.Session() as session:
            stmt = select(func.count()).select_from(PackageModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(PackageModel, key) == value)

            count = session.scalar(stmt)

            return count

    def set_active(self, id: str):
        with self.Session() as session:
            active_stmt = select(PackageModel).where(PackageModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                active_item.active = False

            stmt = select(PackageModel).where(PackageModel.id == id)
            item = session.scalars(stmt).first()

            if item is None:
                return None

            item.active = True
            session.commit()
        return True

    def get_active(self) -> Package:
        with self.Session() as session:
            active_stmt = select(PackageModel).where(PackageModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                return from_row(active_item)
            return None

    def set_active_helper(self, helper: str) -> bool:
        if not helper or helper == "" or helper not in ["numpyhelper", "binaryhelper", "androidhelper"]:
            raise ValueError()

        with self.Session() as session:
            active_stmt = select(PackageModel).where(PackageModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                active_item.helper = helper
                session.commit()
                return True
            item = PackageModel(
                committed_at=datetime.now(),
                description="",
                file_name="",
                helper=helper,
                name="",
                storage_file_name="",
                active=True,
            )

            session.add(item)
            session.commit()

    def delete_active(self) -> bool:
        with self.Session() as session:
            active_stmt = select(PackageModel).where(PackageModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                active_item.active = False
                session.commit()
                return True
            return False
