import pytest
import pymongo

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.tests.stores.helpers.database_helper import mongo_connection, sql_connection, postgres_connection

@pytest.fixture
def test_clients():
    #TODO: SQL versions does not support updated_at field
    # TODO: Create using Client class
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    return [{"client_id":str(uuid.uuid4()), "name":"test_client1", "combiner":"test_combiner", "ip":"121.12.32.2", "combiner_preferred":"", "status":"test_status", 
                   "last_seen":start_date - datetime.timedelta(days=1), "package":"local"},
            {"client_id":str(uuid.uuid4()), "name":"test_client2", "combiner":"test_combiner", "ip":"121.12.32.22", "combiner_preferred":"", "status":"test_status2", 
                   "last_seen":start_date - datetime.timedelta(days=3), "package":"local"},
            {"client_id":str(uuid.uuid4()), "name":"test_client3", "combiner":"test_combiner", "ip":"121.12.32.22", "combiner_preferred":"", "status":"test_status2", 
                   "last_seen":start_date - datetime.timedelta(days=4), "package":"local"},
            {"client_id":str(uuid.uuid4()), "name":"test_client5", "combiner":"test_combiner", "ip":"121.12.32.22", "combiner_preferred":"", "status":"test_status2", 
                   "last_seen":start_date - datetime.timedelta(seconds=5), "package":"remote"},
            {"client_id":str(uuid.uuid4()), "name":"test_client4", "combiner":"test_combiner2", "ip":"121.12.32.22", "combiner_preferred":"", "status":"test_status2", 
                   "last_seen":start_date - datetime.timedelta(weeks=1), "package":"remote"},
            {"client_id":str(uuid.uuid4()), "name":"test_client6", "combiner":"test_combiner2", "ip":"121.12.32.22", "combiner_preferred":"", "status":"test_status2", 
                   "last_seen":start_date - datetime.timedelta(hours=1), "package":"local"},
            {"client_id":str(uuid.uuid4()), "name":"test_client7", "combiner":"test_combiner2", "ip":"121.12.32.22", "combiner_preferred":"", "status":"test_status2", 
                   "last_seen":start_date - datetime.timedelta(minutes=1), "package":"remote"},
                   ]


def test_list_client_store(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_clients):

    for c in test_clients:
        res, _ = mongo_connection.client_store.add(c)
        assert res == True

    for c in test_clients:
        res, _ = postgres_connection.client_store.add(c)
        assert res == True

    for c in test_clients:
        res, _ = sql_connection.client_store.add(c)
        assert res == True

    for name1, db_1, name2, db_2 in [("postgres", postgres_connection, "mongo", mongo_connection), ("sqlite", sql_connection, "mongo", mongo_connection)]:
        print("Running tests between databases {} and {}".format(name1, name2))
        
        sorting_keys = (#None, 
                        "name",
                        "client_id",
                        "last_seen",
                        #"invalid_key"
                        ) 
        limits = (None, 0, 1, 2, 99)
        skips = (None, 0, 1, 2, 99)
        desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)

        opts = list(itertools.product(limits, skips, sorting_keys, desc))
        
        opt_kwargs = ({}, {"ip":"121.12.32.22"}, {"combiner":"test_combiner", "status":"test_status"}, {"name":"test_client1", "status":"test_status2"})

        for kwargs in opt_kwargs:
            for opt in opts:
                print(f"Running tests with options {opt} and kwargs {kwargs}")
                
                res = db_1.client_store.list(*opt, **kwargs)
                count2, gathered_clients2 = res["count"], res["result"]

                res = db_2.client_store.list(*opt, **kwargs)
                count, gathered_clients = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_clients) == len(gathered_clients2)

                for i in range(len(gathered_clients)):
                    assert gathered_clients2[i]["name"] == gathered_clients[i]["name"]
                    assert gathered_clients2[i]["combiner"] == gathered_clients[i]["combiner"]
                    assert gathered_clients2[i]["ip"] == gathered_clients[i]["ip"]
                    # TODO: Should this work?
                    #assert gathered_clients2[i]["combiner_preferred"] == gathered_clients[i]["combiner_preferred"]
                    assert gathered_clients2[i]["status"] == gathered_clients[i]["status"]
                    assert gathered_clients2[i]["last_seen"] == gathered_clients[i]["last_seen"]
                    assert gathered_clients2[i]["package"] == gathered_clients[i]["package"]






    
    
    
    
    
