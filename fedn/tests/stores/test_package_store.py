from typing import List
import pytest

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.package import PackageDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester

@pytest.fixture
def test_packages():
        package1 = PackageDTO(package_id=str(uuid.uuid4()), description="Test package1", file_name="test_package.tar.gz", helper="numpyhelper", name="test_package1")
        package2 = PackageDTO(package_id=str(uuid.uuid4()), description="Test package1", file_name="test_package.tar.gz", helper="numpyhelper", name="test_package2")
        package3 = PackageDTO(package_id=str(uuid.uuid4()), description="Test package1", file_name="test_package.tar.gz", helper="numpyhelper", name="test_package5")
        package4 = PackageDTO(package_id=str(uuid.uuid4()), description="Test package2", file_name="test_package.tar.gz", helper="numpyhelper", name="test_package4")
        package5 = PackageDTO(package_id=str(uuid.uuid4()), description="Test package2", file_name="test_package.tar.gz", helper="numpyhelper", name="test_package7")
        package6 = PackageDTO(package_id=str(uuid.uuid4()), description="Test package2", file_name="test_package.tar.gz", helper="numpyhelper", name="test_package3")
        return [package1, package2, package3, package4, package5, package6]

@pytest.fixture
def test_package():
    return PackageDTO(description="Test package1", file_name="test_package.tar.gz", helper="numpyhelper", name="test_package1")

@pytest.fixture
def options():
    sorting_keys = (None, 
                    "name",
                    "committed_at",
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"description":"Test package2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestPackageStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.package_store
    
    @pytest.fixture
    def test_dto(self, test_package: PackageDTO):
        yield test_package
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_packages: List[PackageDTO]):
        yield from self.helper_prepare_and_clenup(all_db_connections, "package_store", test_packages)

    def test_update(self, store, test_dto):
        with pytest.raises(NotImplementedError):
            store.update(test_dto)
    
    def test_add_delete(self, store, test_dto):
        added_dto = store.add(test_dto)
        added_primary_id = added_dto.primary_id
        assert added_dto is not test_dto, "Added DTO should be a new instance, not the original test DTO."
        assert isinstance(added_dto.primary_id, str)
        assert isinstance(added_dto.committed_at, datetime.datetime)
        assert isinstance(added_dto.updated_at, datetime.datetime)

        added_dto_dict = added_dto.to_dict()
        
        del added_dto_dict[added_dto.primary_key]
        del added_dto_dict["storage_file_name"]
        del added_dto_dict["committed_at"]
        del added_dto_dict["updated_at"]

        test_dto_dict = test_dto.to_dict()
        del test_dto_dict[added_dto.primary_key]
        del test_dto_dict["storage_file_name"]
        del test_dto_dict["committed_at"]
        del test_dto_dict["updated_at"]

        assert added_dto_dict == test_dto_dict 

        # Delete the attribute and check that it is deleted
        success = store.delete(added_primary_id)
        assert success == True

