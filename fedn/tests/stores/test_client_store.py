import pytest
import pymongo

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection


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

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_clients):
    for c in test_clients:
        res, _ = mongo_connection.client_store.add(c)
        assert res == True

    for c in test_clients:
        res, _ = postgres_connection.client_store.add(c)
        assert res == True

    for c in test_clients:
        res, _ = sql_connection.client_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                        "name",
                        "client_id",
                        "last_seen",
                        #"invalid_key"
                        ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"ip":"121.12.32.22"}, {"combiner":"test_combiner", "status":"test_status"}, {"name":"test_client1", "status":"test_status2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestClientStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:                        
                res = db_1.client_store.list(*opt, **kwargs)
                count2, gathered_clients = res["count"], res["result"]

                res = db_2.client_store.list(*opt, **kwargs)
                count, gathered_clients2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_clients) == len(gathered_clients2)

                for i in range(len(gathered_clients)):
                    assert gathered_clients[i]["name"] == gathered_clients2[i]["name"]
                





    
    
    
    
    
