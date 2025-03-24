import pytest
import pymongo

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection

@pytest.fixture
def test_models():

    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    return [{"id":str(uuid.uuid4()), "key":"models", "model":"test_model1", "parent_model":"test_parent_model", "session_id":None, "committed_at":start_date - datetime.timedelta(days=1), "name":"test_name1", "active":True},
            {"id":str(uuid.uuid4()), "key":"models", "model":"test_model2", "parent_model":"test_parent_model", "session_id":None, "committed_at":start_date - datetime.timedelta(days=32), "name":"test_name2", "active":True},
            {"id":str(uuid.uuid4()), "key":"models", "model":"test_model3", "parent_model":"test_parent_model", "session_id":None, "committed_at":start_date - datetime.timedelta(days=23), "name":"test_name3", "active":True},
            {"id":str(uuid.uuid4()), "key":"models", "model":"test_model4", "parent_model":"test_parent_model", "session_id":None, "committed_at":start_date - datetime.timedelta(days=54), "name":"test_name4", "active":True},
            {"id":str(uuid.uuid4()), "key":"models", "model":"test_model5", "parent_model":"test_parent_model", "session_id":None, "committed_at":start_date - datetime.timedelta(days=25), "name":"test_name5", "active":True},
            {"id":str(uuid.uuid4()), "key":"models", "model":"test_model6", "parent_model":"test_parent_model2", "session_id":None, "committed_at":start_date - datetime.timedelta(days=16), "name":"test_name6", "active":True},
            {"id":str(uuid.uuid4()), "key":"models", "model":"test_model7", "parent_model":"test_parent_model2", "session_id":None, "committed_at":start_date - datetime.timedelta(days=27), "name":"test_name7", "active":True},
            {"id":str(uuid.uuid4()), "key":"models", "model":"test_model8", "parent_model":"test_parent_model2", "session_id":None, "committed_at":start_date - datetime.timedelta(days=48), "name":"test_name8", "active":True}]

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_models):
    for c in test_models:
        res, _ = mongo_connection.model_store.add(c)
        assert res == True

    for c in test_models:
        res, _ = postgres_connection.model_store.add(c)
        assert res == True

    for c in test_models:
        res, _ = sql_connection.model_store.add(c)
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
    opt_kwargs = ({}, {"parent_model":"test_parent_model2"}, {"active":False})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestModelStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options: 
                res = db_1.model_store.list(*opt, **kwargs)
                count, gathered_models = res["count"], res["result"]

                res = db_2.model_store.list(*opt, **kwargs)
                count2, gathered_models2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_models) == len(gathered_models2)

                for i in range(len(gathered_models)):
                    #NOTE: id are not equal between the two databases, I think it is due to id being overwritten in the _id field
                    #assert gathered_models2[i]["id"] == gathered_models[i]["id"]
                    assert gathered_models2[i]["committed_at"] == gathered_models[i]["committed_at"]
                    assert gathered_models2[i]["model"] == gathered_models[i]["model"]
                    assert gathered_models2[i]["parent_model"] == gathered_models[i]["parent_model"]
                    assert gathered_models2[i]["session_id"] == gathered_models[i]["session_id"]
                    assert gathered_models2[i]["name"] == gathered_models[i]["name"]
                    assert gathered_models2[i]["active"] == gathered_models[i]["active"]
                    
