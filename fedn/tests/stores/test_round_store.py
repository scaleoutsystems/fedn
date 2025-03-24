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

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_rounds):
    for c in test_rounds:
        res, _ = mongo_connection.round_store.add(c)
        assert res == True

    for c in test_rounds:
        res, _ = postgres_connection.round_store.add(c)
        assert res == True

    for c in test_rounds:
        res, _ = sql_connection.round_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                        "status", 
                        #"invalid_key"
                        ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"status":"test_status4,test_status5"}, {"status":"blah"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestRoundStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                res = db_1.round_store.list(*opt, **kwargs)
                count, gathered_rounds = res["count"], res["result"]

                res = db_2.round_store.list(*opt, **kwargs)
                count2, gathered_rounds2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_rounds) == len(gathered_rounds2)

                for i in range(len(gathered_rounds)):
                    assert gathered_rounds2[i]["round_id"] == gathered_rounds[i]["round_id"]
                    assert gathered_rounds2[i]["status"] == gathered_rounds[i]["status"]


