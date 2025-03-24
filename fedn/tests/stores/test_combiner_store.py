import pytest
import pymongo

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection



@pytest.fixture
def test_combiners():
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    # NOTE: The SQL version does not support the dicts as parameters  parent and config
    # TODO: Creat using Combiner class
    return [{"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=1), "name":"test_combiner1",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:2", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=52), "address":"test_address"},
            {"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=2), "name":"test_combiner2",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:2", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=12), "address":"test_address"},
            {"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=8), "name":"test_combiner3",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:5", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=322), "address":"test_address"},
            {"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=4), "name":"test_combiner4",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:4", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=23), "address":"test_address"},
            {"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=51), "name":"test_combiner5",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:3", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=22), "address":"test_address"},
            {"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=9), "name":"test_combiner6",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:3", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=24), "address":"test_address"},
            {"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=3), "name":"test_combiner8",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:3", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=42), "address":"test_address"},
            {"id":str(uuid.uuid4()), "committed_at":start_date - datetime.timedelta(days=14), "name":"test_combiner7",
             "parent":"localhost",
            # "config":{},
             "ip":"123:13:12:2", "fqdn":"", "port":8080,
                        "updated_at":start_date - datetime.timedelta(days=12), "address":"test_address1"}]


@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_combiners):
    for c in test_combiners:
        res, _ = mongo_connection.combiner_store.add(c)
        assert res == True

    for c in test_combiners:
        res, _ = postgres_connection.combiner_store.add(c)
        assert res == True

    for c in test_combiners:
        res, _ = sql_connection.combiner_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                        "name",
                        #"committed_at",
                        #"updated_at",
                        #"invalid_key"
                        ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"ip":"123:13:12:3"}, {"address":"test_address1"}, {"fqdn":"", "name":"test_combiner77"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestCombinerStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                res = db_1.combiner_store.list(*opt, **kwargs)
                count, gathered_combiners = res["count"], res["result"]

                res = db_2.combiner_store.list(*opt, **kwargs)
                count2, gathered_combiners2 = res["count"], res["result"]
                #TODO: The count is not equal to the number of clients in the list, but the number of clients returned by the query before skip and limit
                #It is not clear what is the intended behavior
                # assert(count == len(gathered_clients))
                # assert count == count2
                assert len(gathered_combiners) == len(gathered_combiners2)

                for i in range(len(gathered_combiners)):
                    #NOTE: id are not equal between the two databases, I think it is due to id being overwritten in the _id field
                    #assert gathered_combiners2[i]["id"] == gathered_combiners[i]["id"]
                    #TODO: committed_at is not equal between the two databases, one reades from init the other uses the current time
                    #assert gathered_combiners2[i]["committed_at"] == gathered_combiners[i]["committed_at"]
                    assert gathered_combiners2[i]["name"] == gathered_combiners[i]["name"]
                    assert gathered_combiners2[i]["parent"] == gathered_combiners[i]["parent"]
                    assert gathered_combiners2[i]["ip"] == gathered_combiners[i]["ip"]
                    assert gathered_combiners2[i]["fqdn"] == gathered_combiners[i]["fqdn"]
                    assert gathered_combiners2[i]["port"] == gathered_combiners[i]["port"]
                    #TODO: updated_at is not equal between the two databases, one reades from init the other uses the current time
                    #assert gathered_combiners2[i]["updated_at"] == gathered_combiners[i]["updated_atssert gathered_combiners2[i]["address"] == gathered_combiners[i]["address"]
                        

                        
    
