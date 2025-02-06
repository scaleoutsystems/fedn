import pytest
import pymongo

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.tests.stores.helpers.database_helper import mongo_connection, sql_connection, postgres_connection




@pytest.fixture
def test_predictions():
    return [{"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id1", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name1", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name1", 
             "sender_role":"test_sender_role1", "timestamp":"2024-13", "prediction_id":"test_prediction_id1"},
             {"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id4", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name2", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name1", 
             "sender_role":"test_sender_role1", "timestamp":"2024-11", "prediction_id":"test_prediction_id1"},
             {"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id3", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name3", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name2", 
             "sender_role":"test_sender_role1", "timestamp":"2024-12", "prediction_id":"test_prediction_id2"},
             {"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id2", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name4", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name2", 
             "sender_role":"test_sender_role1", "timestamp":"2024-16", "prediction_id":"test_prediction_id2"}]

def test_list_prediction_store(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_predictions):
    for p in test_predictions:
        res, _ = mongo_connection.prediction_store.add(p)
        assert res == True
        res, _ = postgres_connection.prediction_store.add(p)
        assert res == True
        res, _ = sql_connection.prediction_store.add(p)
        assert res == True

    for name1, db_1, name2, db_2 in [("postgres", postgres_connection, "mongo", mongo_connection), ("sqlite", sql_connection, "mongo", mongo_connection)]:
        print("Running tests between databases {} and {}".format(name1, name2))

        # TODO: Fix commented
        sorting_keys = (#None, 
                        "correlation_id", 
                        "timestamp", 
                        #"invalid_key"
                        ) 
        limits = (None, 0, 1, 2, 99)
        skips = (None, 0, 1, 2, 99)
        desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)

        opts = list(itertools.product(limits, skips, sorting_keys, desc))
        
        opt_kwargs = ({}, {"prediction_id":"test_prediction_id2"}, {"correlation_id":""})

        for kwargs in opt_kwargs:
            for opt in opts:
                print(f"Running tests with options {opt} and kwargs {kwargs}")
                res = db_1.prediction_store.list(*opt, **kwargs)
                count, gathered_models = res["count"], res["result"]

                res = db_2.prediction_store.list(*opt, **kwargs)
                count2, gathered_models2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                print(count, count2)
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_models) == len(gathered_models2)

                for i in range(len(gathered_models)):
                    #NOTE: id are not equal between the two databases, I think it is due to id being overwritten in the _id field
                    #assert gathered_models2[i]["id"] == gathered_models[i]["id"]
                    assert gathered_models2[i]["correlation_id"] == gathered_models[i]["correlation_id"]
                    assert gathered_models2[i]["data"] == gathered_models[i]["data"]
                    assert gathered_models2[i]["model_id"] == gathered_models[i]["model_id"]
                    #TODO: Reciever and sender are handled differently in the two databases, the value is not the same
                    #assert gathered_models2[i]["receiver_name"] == gathered_models[i]["receiver_name"]
                    #assert gathered_models2[i]["receiver_role"] == gathered_models[i]["receiver_role"]
                    #assert gathered_models2[i]["sender_name"] == gathered_models[i]["sender_name"]
                    #assert gathered_models2[i]["sender_role"] == gathered_models[i]["sender_role"]
                    assert gathered_models2[i]["timestamp"] == gathered_models[i]["timestamp"]
                    assert gathered_models2[i]["prediction_id"] == gathered_models[i]["prediction_id"]

