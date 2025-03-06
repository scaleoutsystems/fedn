import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple, Union

import pymongo
from pymongo.database import Database
from sqlalchemy import select
from werkzeug.utils import secure_filename

from fedn.network.storage.statestore.stores.dto import PackageDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import PackageModel, from_orm_model


def is_active_package(data: dict, active_package: dict) -> bool:
    if active_package and "package_id" in active_package and "package_id" in data:
        return active_package["package_id"] == data["package_id"]
    return False


def _complement_with_storage_filename(item: Dict):
    extension = item["file_name"].rsplit(".", 1)[1].lower()

    if "storage_file_name" not in item or item["storage_file_name"] is None:
        storage_file_name = secure_filename(f"{str(uuid.uuid4())}.{extension}")
        item["storage_file_name"] = storage_file_name


class PackageStore(Store[PackageDTO]):
    @abstractmethod
    def set_active(self, id: str) -> bool:
        """Set the active entity
        param id: The id of the entity
            type: str
        return: Whether the operation was successful
        """
        pass

    @abstractmethod
    def get_active(self) -> PackageDTO:
        """Get the active entity
        return: The entity
        """
        pass

    @abstractmethod
    def set_active_helper(self, helper: str) -> bool:
        """Set the active helper
        param helper: The helper to set as active
            type: str
        return: Whether the operation was successful
        """
        pass

    @abstractmethod
    def delete_active(self) -> bool:
        """Delete the active entity
        return: Whether the operation was successful
        """
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


def validate(item: Dict) -> Tuple[bool, str]:
    if "file_name" not in item or not item["file_name"]:
        return False, "File name is required"

    if not allowed_file_extension(item["file_name"]):
        return False, "File extension not allowed"

    if "helper" not in item or not validate_helper(item["helper"]):
        return False, "Helper is required"

    return True, ""


def validate_helper(helper: str) -> bool:
    if not helper or helper == "" or helper not in ["numpyhelper", "binaryhelper", "androidhelper"]:
        return False
    return True


class MongoDBPackageStore(PackageStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "package_id")

    def get(self, id: str) -> PackageDTO:
        document = self.database[self.collection].find_one({self.primary_key: id})

        if document is None:
            return None

        response_active = self.database[self.collection].find_one({"key": "active"})
        active = is_active_package(document, response_active)
        document["active"] = active

        return self._dto_from_document(document)

    def _complement(self, item: Dict):
        if "key" not in item or item["key"] is None:
            item["key"] = "package_trail"

        _complement_with_storage_filename(item)

    def set_active(self, id: str) -> bool:
        kwargs = {self.primary_key: id}
        kwargs["key"] = "package_trail"

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            return False

        # TODO: Use seperate table to store active package
        activePackage = PackageDTO()
        activePackage.patch(document, throw_on_extra_keys=False)
        activePackage.active = True
        activePackage.committed_at = datetime.now()
        obj_to_insert = {"key": "active", **activePackage.to_db()}

        self.database[self.collection].update_one({"key": "active"}, {"$set": obj_to_insert}, upsert=True)

        return True

    def get_active(self) -> PackageDTO:
        kwargs = {"key": "active"}
        document = self.database[self.collection].find_one(kwargs)
        if document is None:
            return None
        document["active"] = True
        return self._dto_from_document(document)

    def set_active_helper(self, helper: str) -> bool:
        if not validate_helper(helper):
            raise ValueError()
        try:
            self.database[self.collection].update_one({"key": "active"}, {"$set": {"helper": helper}}, upsert=True)
        except Exception:
            return False

    def update(self, item: PackageDTO):
        raise NotImplementedError("Update not implemented for PackageStore")

    def add(self, item: PackageDTO) -> Tuple[bool, Union[str, PackageDTO]]:
        item_dict = item.to_db(exclude_unset=False)
        valid, message = validate(item_dict)
        if not valid:
            return False, message

        self._complement(item_dict)

        success, obj = self.mongo_add(item_dict)
        if not success:
            return False, obj
        return True, self._dto_from_document(obj)

    def delete(self, id: str) -> bool:
        kwargs = {self.primary_key: id, "key": "package_trail"}
        result = self.database[self.collection].delete_one(kwargs).deleted_count == 1
        if not result:
            return False

        # Remove Active Package if it is the one being deleted
        kwargs["key"] = "active"
        document_active = self.database[self.collection].find_one(kwargs)
        if document_active is not None:
            return self.database[self.collection].delete_one(kwargs).deleted_count == 1
        return True

    def delete_active(self) -> bool:
        kwargs = {"key": "active"}
        return self.database[self.collection].delete_one(kwargs).deleted_count == 1

    def select(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs) -> List[PackageDTO]:
        kwargs["key"] = "package_trail"
        result = self.mongo_select(limit, skip, sort_key, sort_order, **kwargs)
        return [self._dto_from_document(item) for item in result]

    def list(
        self,
        limit: int,
        skip: int,
        sort_key: str,
        sort_order=pymongo.DESCENDING,
        **kwargs,
    ):
        raise NotImplementedError

    def count(self, **kwargs) -> int:
        kwargs["key"] = "package_trail"
        return self.mongo_count(**kwargs)

    def _dto_from_document(self, document: Dict) -> PackageDTO:
        item_dict = from_document(document)
        return PackageDTO().populate_with(item_dict)


class SQLPackageStore(PackageStore, SQLStore[PackageModel]):
    def __init__(self, Session):
        super().__init__(Session, PackageModel)

    def add(self, item: PackageDTO) -> Tuple[bool, Union[str, PackageDTO]]:
        item_dict = item.to_db(exclude_unset=False)
        valid, message = validate(item_dict)
        if not valid:
            return False, message

        item_dict = self._orm_dict_from_dto_dict(item_dict)

        with self.Session() as session:
            package = PackageModel(**item_dict)
            success, obj = self.sql_add(session, package)
            if not success:
                return False, obj
            return True, self._dto_from_orm_model(obj)

    def get(self, id: str) -> PackageDTO:
        with self.Session() as session:
            item = self.sql_get(session, id)
            if item is None:
                return None
            return self._dto_from_orm_model(item)

    def update(self, item: PackageDTO):
        raise NotImplementedError

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def select(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            result = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(item) for item in result]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        raise NotImplementedError

    def count(self, **kwargs):
        return self.sql_count(**kwargs)

    def set_active(self, id: str):
        with self.Session() as session:
            active_stmt = select(PackageModel).where(PackageModel.active)
            active_item = session.scalars(active_stmt).first()

            stmt = select(PackageModel).where(PackageModel.id == id)
            new_active_item = session.scalars(stmt).first()
            if new_active_item is None:
                return False

            new_active_item.active = True
            if active_item:
                active_item.active = False
            session.commit()
        return True

    def get_active(self) -> PackageDTO:
        with self.Session() as session:
            active_stmt = select(PackageModel).where(PackageModel.active)
            active_item = session.scalars(active_stmt).first()
            if active_item:
                return self._dto_from_orm_model(active_item)
            return None

    def set_active_helper(self, helper: str) -> bool:
        if not validate_helper(helper):
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

    def _orm_dict_from_dto_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("package_id")
        _complement_with_storage_filename(item_dict)
        return item_dict

    def _dto_from_orm_model(self, item: PackageModel) -> PackageDTO:
        orm_dict = from_orm_model(item, PackageModel)
        orm_dict["package_id"] = orm_dict.pop("id")
        return PackageDTO().populate_with(orm_dict)
