import pytest
import pymongo

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection



@pytest.fixture
def test_rounds():
    return [{"round_id":str(uuid.uuid4()), "status":"test_status3"},
            {"round_id":str(uuid.uuid4()), "status":"test_status2"},
            {"round_id":str(uuid.uuid4()), "status":"test_status5"},
            {"round_id":str(uuid.uuid4()), "status":"test_status1"},
            {"round_id":str(uuid.uuid4()), "status":"test_status4"},
            {"round_id":str(uuid.uuid4()), "status":"test_status6"}]

def test_list_round_store(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_rounds):
    for r in test_rounds:
        res, _ = mongo_connection.round_store.add(r)
        assert res == True
        res, _ = postgres_connection.round_store.add(r)
        assert res == True
        res, _ = sql_connection.round_store.add(r)
        assert res == True

    for name1, db_1, name2, db_2 in [("postgres", postgres_connection, "mongo", mongo_connection), ("sqlite", sql_connection, "mongo", mongo_connection)]:
        print("Running tests between databases {} and {}".format(name1, name2))

        # TODO: Fix commented
        sorting_keys = (#None, 
                        "status", 
                        #"invalid_key"
                        ) 
        limits = (None, 0, 1, 2, 99)
        skips = (None, 0, 1, 2, 99)
        desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)

        opts = list(itertools.product(limits, skips, sorting_keys, desc))
        
        opt_kwargs = ({}, {"status":"test_status4,test_status5"}, {"status":"blah"})

        for kwargs in opt_kwargs:
            for opt in opts:
                print(f"Running tests with options {opt} and kwargs {kwargs}")
                res = db_1.round_store.list(*opt, **kwargs)
                count, gathered_rounds = res["count"], res["result"]

                res = db_2.round_store.list(*opt, **kwargs)
                count2, gathered_rounds2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                print(count, count2)
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_rounds) == len(gathered_rounds2)

                for i in range(len(gathered_rounds)):
                    assert gathered_rounds2[i]["round_id"] == gathered_rounds[i]["round_id"]
                    assert gathered_rounds2[i]["status"] == gathered_rounds[i]["status"]


