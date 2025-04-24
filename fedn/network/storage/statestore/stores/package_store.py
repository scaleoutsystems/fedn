import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Dict, List

from pymongo.database import Database
from sqlalchemy import select
from werkzeug.utils import secure_filename

from fedn.network.storage.statestore.stores.dto import PackageDTO
from fedn.network.storage.statestore.stores.dto.package import validate_helper
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import PackageModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


def is_active_package(package_id, active_package: dict) -> bool:
    if active_package and "package_id" in active_package and package_id:
        return active_package["package_id"] == package_id
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


class MongoDBPackageStore(PackageStore, MongoDBStore[PackageDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "package_id")

    def get(self, id: str) -> PackageDTO:
        package = super().get(id)
        if package is None:
            return None

        response_active = self.database[self.collection].find_one({"key": "active"})
        active = is_active_package(package.package_id, response_active)
        package.active = active

        return package

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

    def list(self, limit=0, skip=0, sort_key=None, sort_order=SortOrder.DESCENDING, **kwargs) -> List[PackageDTO]:
        kwargs["key"] = "package_trail"
        packages = super().list(limit, skip, sort_key, sort_order, **kwargs)

        if packages is None or len(packages) == 0:
            return []

        active_package = self.get_active()

        if active_package:
            for package in packages:
                package.active = is_active_package(package.package_id, active_package.to_dict())

        return packages

    def count(self, **kwargs) -> int:
        kwargs["key"] = "package_trail"
        return super().count(**kwargs)

    def set_active(self, id: str) -> bool:
        kwargs = {self.primary_key: id}
        kwargs["key"] = "package_trail"

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            return False

        # TODO: Use seperate table to store active package
        activePackage = PackageDTO()
        activePackage.patch_with(document, throw_on_extra_keys=False)
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
            raise ValueError
        try:
            self.database[self.collection].update_one({"key": "active"}, {"$set": {"helper": helper}}, upsert=True)
        except Exception:
            return False

    def _document_from_dto(self, item: PackageDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        item_dict["key"] = "package_trail"
        _complement_with_storage_filename(item_dict)
        return item_dict

    def _dto_from_document(self, document: Dict) -> PackageDTO:
        item_dict = from_document(document)
        del item_dict["key"]
        return PackageDTO().patch_with(item_dict, throw_on_extra_keys=False)


class SQLPackageStore(PackageStore, SQLStore[PackageDTO, PackageModel]):
    def __init__(self, Session):
        super().__init__(Session, PackageModel, "package_id")

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

    def _update_orm_model_from_dto(self, entity: PackageModel, item: PackageDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("package_id", None)
        _complement_with_storage_filename(item_dict)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: PackageModel) -> PackageDTO:
        orm_dict = from_orm_model(item, PackageModel)
        orm_dict["package_id"] = orm_dict.pop("id")
        return PackageDTO().populate_with(orm_dict)
