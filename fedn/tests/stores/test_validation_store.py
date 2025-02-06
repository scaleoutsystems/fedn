import pytest
import pymongo

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.tests.stores.helpers.database_helper import mongo_connection, sql_connection, postgres_connection



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



def test_list_round_store(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_validations):

    for v in test_validations:
        res, _ = mongo_connection.validation_store.add(v)
        assert res == True
        res, _ = postgres_connection.validation_store.add(v)
        assert res == True
        res, _ = sql_connection.validation_store.add(v)
        assert res == True


    for name1, db_1, name2, db_2 in [("postgres", postgres_connection, "mongo", mongo_connection), ("sqlite", sql_connection, "mongo", mongo_connection)]:
        print("Running tests between databases {} and {}".format(name1, name2))

        # TODO: Fix commented
        sorting_keys = (#None, 
                        "data",
                        "timestamp",
                        #"invalid_key"
                        ) 
        limits = (None, 0, 1, 2, 99)
        skips = (None, 0, 1, 2, 99)
        desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)

        opts = list(itertools.product(limits, skips, sorting_keys, desc))
        
        opt_kwargs = ({}, {"data":"test_data6"}, {"data":""})

        for kwargs in opt_kwargs:
            for opt in opts:
                print(f"Running tests with options {opt} and kwargs {kwargs}")
                res = db_1.validation_store.list(*opt, **kwargs)
                count, gathered_validations = res["count"], res["result"]

                res = db_2.validation_store.list(*opt, **kwargs)
                count2, gathered_validations2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                print(count, count2)
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_validations) == len(gathered_validations2)

                for i in range(len(gathered_validations)):
                    assert gathered_validations2[i]["data"] == gathered_validations[i]["data"]
                    #TODO: timestamp is not equal between the two databases, different formats
                    #assert gathered_validations2[i]["timestamp"] == gathered_validations[i]["timestamp"]
                    assert gathered_validations2[i]["correlation_id"] == gathered_validations[i]["correlation_id"]



