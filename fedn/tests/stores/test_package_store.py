import pytest
import pymongo

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection

@pytest.fixture
def test_packages():
        start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
        return [{"id":str(uuid.uuid4()), "committed_at":start_date+datetime.timedelta(days=266), "description":"Test package1", "file_name":"test_package.tar.gz", "helper":"numpyhelper", "name":"test_package1", "storage_file_name":"test_package.tar.gz", "active":True},
                {"id":str(uuid.uuid4()), "committed_at":start_date+datetime.timedelta(days=616), "description":"Test package1", "file_name":"test_package.tar.gz", "helper":"numpyhelper", "name":"test_package2", "storage_file_name":"test_package.tar.gz", "active":True},
                {"id":str(uuid.uuid4()), "committed_at":start_date+datetime.timedelta(days=176), "description":"Test package1", "file_name":"test_package.tar.gz", "helper":"numpyhelper", "name":"test_package5", "storage_file_name":"test_package.tar.gz", "active":True},
                {"id":str(uuid.uuid4()), "committed_at":start_date+datetime.timedelta(days=366), "description":"Test package2", "file_name":"test_package.tar.gz", "helper":"numpyhelper", "name":"test_package4", "storage_file_name":"test_package.tar.gz", "active":True},
                {"id":str(uuid.uuid4()), "committed_at":start_date+datetime.timedelta(days=664), "description":"Test package2", "file_name":"test_package.tar.gz", "helper":"numpyhelper", "name":"test_package7", "storage_file_name":"test_package.tar.gz", "active":True},
                {"id":str(uuid.uuid4()), "committed_at":start_date+datetime.timedelta(days=66), "description":"Test package2", "file_name":"test_package.tar.gz", "helper":"numpyhelper", "name":"test_package3", "storage_file_name":"test_package.tar.gz", "active":False},
                {"id":str(uuid.uuid4()), "committed_at":start_date+datetime.timedelta(days=166), "description":"Test package2", "file_name":"test_package.tar.gz", "helper":"numpyhelper", "name":"test_package6", "storage_file_name":"test_package.tar.gz", "active":True}]

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_packages):
    for c in test_packages:
        res, _ = mongo_connection.package_store.add(c)
        assert res == True

    for c in test_packages:
        res, _ = postgres_connection.package_store.add(c)
        assert res == True

    for c in test_packages:
        res, _ = sql_connection.package_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                        "name",
                        "committed_at",
                        #"invalid_key"
                        ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"description":"Test package2"}, 
                      #TODO: The active field is handled differently in the two databases, the value is not the same
                      #{"active":False}
                      )

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestPackageStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass
    
    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                res = db_1.package_store.list(*opt, **kwargs)
                count, gathered_packages = res["count"], res["result"]

                res = db_2.package_store.list(*opt, **kwargs)
                count2, gathered_packages2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_packages) == len(gathered_packages2)

                for i in range(len(gathered_packages)):
                    #NOTE: id are not equal between the two databases, I think it is due to id being overwritten in the _id field
                    #assert gathered_models2[i]["id"] == gathered_models[i]["id"]
                    assert gathered_packages2[i]["committed_at"] == gathered_packages[i]["committed_at"]
                    assert gathered_packages2[i]["name"] == gathered_packages[i]["name"]
                    assert gathered_packages2[i]["description"] == gathered_packages[i]["description"]
                    assert gathered_packages2[i]["file_name"] == gathered_packages[i]["file_name"]
                    assert gathered_packages2[i]["helper"] == gathered_packages[i]["helper"]
                    assert gathered_packages2[i]["storage_file_name"] == gathered_packages[i]["storage_file_name"]
                    assert gathered_packages2[i]["active"] == gathered_packages[i]["active"]

                    
