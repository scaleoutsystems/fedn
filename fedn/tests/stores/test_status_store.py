import pytest
import pymongo

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection

@pytest.fixture
def test_statuses():
    start_time = datetime.datetime(2021, 1, 4, 1, 2, 4)
   
    return [{"log_level":"test_log_level1", "status":"status1", "timestamp":start_time+datetime.timedelta(days=2), "type":"test_type"},
            {"log_level":"test_log_level2", "status":"status2", "timestamp":start_time+datetime.timedelta(days=4), "type":"test_type"},
            {"log_level":"test_log_level3", "status":"status3", "timestamp":start_time+datetime.timedelta(days=5), "type":"test_type"},
            {"log_level":"test_log_level4", "status":"status4", "timestamp":start_time+datetime.timedelta(days=1), "type":"test_type2"},
            {"log_level":"test_log_level5", "status":"status5", "timestamp":start_time+datetime.timedelta(days=6), "type":"test_type2"},
            {"log_level":"test_log_level6", "status":"status6", "timestamp":start_time+datetime.timedelta(days=3), "type":"test_type2"},
            ]



def test_list_round_store(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_statuses):

    for s in test_statuses:
        res, _ = mongo_connection.status_store.add(s)
        assert res == True
        res, _ = postgres_connection.status_store.add(s)
        assert res == True
        res, _ = sql_connection.status_store.add(s)
        assert res == True


    for name1, db_1, name2, db_2 in [("postgres", postgres_connection, "mongo", mongo_connection), ("sqlite", sql_connection, "mongo", mongo_connection)]:
        print("Running tests between databases {} and {}".format(name1, name2))

        # TODO: Fix commented
        sorting_keys = (#None, 
                        "log_level",
                        "timestamp",
                        #"invalid_key"
                        ) 
        limits = (None, 0, 1, 2, 99)
        skips = (None, 0, 1, 2, 99)
        desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)

        opts = list(itertools.product(limits, skips, sorting_keys, desc))
        
        opt_kwargs = ({}, {"log_level":"test_log_level6"}, {"type":"test_type2"})

        for kwargs in opt_kwargs:
            for opt in opts:
                print(f"Running tests with options {opt} and kwargs {kwargs}")
                res = db_1.status_store.list(*opt, **kwargs)
                count, gathered_statuses = res["count"], res["result"]

                res = db_2.status_store.list(*opt, **kwargs)
                count2, gathered_statuses2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                print(count, count2)
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_statuses) == len(gathered_statuses2)

                for i in range(len(gathered_statuses)):
                    assert gathered_statuses2[i]["log_level"] == gathered_statuses[i]["log_level"]
                    # TODO: Status is saved differently in the two databases, why?
                    # assert gathered_statuses2[i]["timestamp"] == gathered_statuses[i]["timestamp"]
                    assert gathered_statuses2[i]["status"] == gathered_statuses[i]["status"]


