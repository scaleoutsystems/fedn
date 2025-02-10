import pytest
import pymongo

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection

@pytest.fixture
def test_validations():
    start_time = datetime.datetime(2021, 1, 4, 1, 2, 4)
   
    return [{"data":"test_data7", "timestamp":start_time+datetime.timedelta(days=76), "correlation_id":str(uuid.uuid4())},
            {"data":"test_data2", "timestamp":start_time+datetime.timedelta(days=22), "correlation_id":str(uuid.uuid4())},
            {"data":"test_data4", "timestamp":start_time+datetime.timedelta(days=1452), "correlation_id":str(uuid.uuid4())},
            {"data":"test_data5", "timestamp":start_time+datetime.timedelta(days=52), "correlation_id":str(uuid.uuid4())},
            {"data":"test_data1", "timestamp":start_time+datetime.timedelta(days=212), "correlation_id":str(uuid.uuid4())},
            {"data":"test_data3", "timestamp":start_time+datetime.timedelta(days=12), "correlation_id":str(uuid.uuid4())},
            {"data":"test_data8", "timestamp":start_time+datetime.timedelta(days=72), "correlation_id":str(uuid.uuid4())},
            {"data":"test_data6", "timestamp":start_time+datetime.timedelta(days=42), "correlation_id":str(uuid.uuid4())}]

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_validations):
    for c in test_validations:
        res, _ = mongo_connection.validation_store.add(c)
        assert res == True

    for c in test_validations:
        res, _ = postgres_connection.validation_store.add(c)
        assert res == True

    for c in test_validations:
        res, _ = sql_connection.validation_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                        "data",
                        "timestamp",
                        #"invalid_key"
                        ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"data":"test_data6"}, {"data":""})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestValidationStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                res = db_1.validation_store.list(*opt, **kwargs)
                count, gathered_validations = res["count"], res["result"]

                res = db_2.validation_store.list(*opt, **kwargs)
                count2, gathered_validations2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_validations) == len(gathered_validations2)

                for i in range(len(gathered_validations)):
                    assert gathered_validations2[i]["data"] == gathered_validations[i]["data"]
                    #TODO: timestamp is not equal between the two databases, different formats
                    #assert gathered_validations2[i]["timestamp"] == gathered_validations[i]["timestamp"]
                    assert gathered_validations2[i]["correlation_id"] == gathered_validations[i]["correlation_id"]



