import pytest

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.package import PackageDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

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
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_packages):
    for c in test_packages:
        mongo_connection.package_store.add(c)
        postgres_connection.package_store.add(c)
        sql_connection.package_store.add(c)

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_packages:
        mongo_connection.package_store.delete(c.package_id)
        postgres_connection.package_store.delete(c.package_id)
        sql_connection.package_store.delete(c.package_id)



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

class TestPackageStore:
    
    def test_add_update_delete(self, db_connection: DatabaseConnection, test_package: PackageDTO):
        test_package.check_validity()

        # Add a package and check that we get the added package back
        read_package1 = db_connection.package_store.add(test_package)
        assert isinstance(read_package1.package_id, str)
        assert isinstance(read_package1.committed_at, datetime.datetime)
        assert isinstance(read_package1.storage_file_name, str)

        read_package1_dict = read_package1.to_dict()
        package_id = read_package1_dict["package_id"]
        del read_package1_dict["package_id"]
        del read_package1_dict["committed_at"]
        del read_package1_dict["storage_file_name"]

        test_package_dict = test_package.to_dict()
        del test_package_dict["package_id"]
        del test_package_dict["committed_at"]
        del test_package_dict["storage_file_name"]

        assert read_package1_dict == test_package_dict

        # Assert we get the same package back
        read_package2 = db_connection.package_store.get(package_id)
        assert read_package2 is not None
        assert read_package2.to_dict() == read_package1.to_dict()    

        # Delete the package and check that it is deleted
        success = db_connection.package_store.delete(package_id)
        assert success == True
    
    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_packages = db_1.package_store.list(*opt, **kwargs)
                count = db_1.package_store.count(**kwargs)
                
                gathered_packages2 = db_2.package_store.list(*opt, **kwargs)
                count2 = db_2.package_store.count(**kwargs)
                
                assert count == count2
                assert len(gathered_packages) == len(gathered_packages2)

                for i in range(len(gathered_packages)):
                    assert gathered_packages2[i].package_id == gathered_packages[i].package_id
                    
                    
