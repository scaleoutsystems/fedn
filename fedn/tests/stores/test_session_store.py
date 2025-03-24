import pytest
import pymongo

import time
import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection


@pytest.fixture
def test_sessions():
    start_time = datetime.datetime(2021, 1, 4, 1, 2, 4)
    
    model = {"id":str(uuid.uuid4()), "key":"models", "model":str(uuid.uuid4()), "parent_model":"test_parent_model", "session_id":None, "committed_at":start_time, "name":"test_name1", "active":True}

    session_config = {"aggregator":"test_aggregator", "round_timeout":100, "buffer_size":100, "delete_models_storage":True, 
                      "clients_required":10, "validate":True, "helper_type":"test_helper_type", "model_id":model["model"]}

    return model,[{"id":str(uuid.uuid4()),"name":"sessionname1",  "session_config":session_config},
            {"id":str(uuid.uuid4()),"name":"sessionname3",  "session_config":session_config},
            {"id":str(uuid.uuid4()),"name":"sessionname5",  "session_config":session_config},
            {"id":str(uuid.uuid4()),"name":"sessionname4",  "session_config":session_config},
            {"id":str(uuid.uuid4()),"name":"sessionname6",  "session_config":session_config},
            {"id":str(uuid.uuid4()),"name":"sessionname2",   "session_config":session_config},
            ]

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_sessions):
    model, test_sessions = test_sessions
    res, _ = mongo_connection.model_store.add(model)
    assert res == True
    res, _ = postgres_connection.model_store.add(model)
    assert res == True
    res, _ = sql_connection.model_store.add(model)
    assert res == True

    for c in test_sessions:
        res, _ = mongo_connection.session_store.add(c)
        assert res == True

    for c in test_sessions:
        res, _ = postgres_connection.session_store.add(c)
        assert res == True

    for c in test_sessions:
        res, _ = sql_connection.session_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                    "name",
                    #"committed_at",
                    #"invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"status":"test_status4"}, {"status":"blah"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestSessionStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass


    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                res = db_1.session_store.list(*opt, **kwargs)
                count, gathered_sessions = res["count"], res["result"]

                res = db_2.session_store.list(*opt, **kwargs)
                count2, gathered_sessions2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_sessions) == len(gathered_sessions2)

                for i in range(len(gathered_sessions)):
                    #assert gathered_sessions2[i]["id"] == gathered_sessions[i]["id"]
                    assert gathered_sessions2[i]["name"] == gathered_sessions[i]["name"]


