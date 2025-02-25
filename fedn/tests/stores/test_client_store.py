import pytest
import pymongo

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import Client

@pytest.fixture
def test_clients():
    #TODO: SQL versions does not support updated_at field
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    return [
        Client(client_id=str(uuid.uuid4()), name="test_client1", combiner="test_combiner", ip="121.12.32.2", combiner_preferred="", status="test_status", last_seen=start_date - datetime.timedelta(days=1), package="local"),
        Client(client_id=str(uuid.uuid4()), name="test_client2", combiner="test_combiner", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(days=3), package="local"),
        Client(client_id=str(uuid.uuid4()), name="test_client3", combiner="test_combiner", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(days=4), package="local"),
        Client(client_id=str(uuid.uuid4()), name="test_client5", combiner="test_combiner", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(seconds=5), package="remote"),
        Client(client_id=str(uuid.uuid4()), name="test_client4", combiner="test_combiner2", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(weeks=1), package="remote"),
        Client(client_id=str(uuid.uuid4()), name="test_client6", combiner="test_combiner2", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(hours=1), package="local"),
        Client(client_id=str(uuid.uuid4()), name="test_client7", combiner="test_combiner2", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(minutes=1), package="remote"),
    ]


@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_clients):
    for c in test_clients:
        res, msg = mongo_connection.client_store.add(c)
        assert res == True

    for c in test_clients:
        res, _ = postgres_connection.client_store.add(c)
        assert res == True

    for c in test_clients:
        res, _ = sql_connection.client_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_clients:
        res = mongo_connection.client_store.delete(c.client_id)
        assert res == True
    
    for c in test_clients:
        res = postgres_connection.client_store.delete(c.client_id)
        assert res == True
    
    for c in test_clients:
        res = sql_connection.client_store.delete(c.client_id)
        assert res == True



@pytest.fixture
def options():
    sorting_keys = (None,
                    "name",
                    "client_id",
                    "last_seen",
                    "ip", # None unique key 
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"ip":"121.12.32.22"}, {"combiner":"test_combiner", "status":"test_status"}, {"name":"test_client1", "status":"test_status2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestClientStore:

    def test_add_get_update_delete_postgres(self, postgres_connection:DatabaseConnection):
        self.helper_add_get_update_delete(postgres_connection)
    
    def test_add_get_update_delete_sqlite(self, sql_connection:DatabaseConnection):
        self.helper_add_get_update_delete(sql_connection)
    
    def test_add_get_update_delete_mongo(self, mongo_connection:DatabaseConnection):
        self.helper_add_get_update_delete(mongo_connection)

    def helper_add_get_update_delete(self, db:DatabaseConnection):
        start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
        c = Client(name = "name", combiner = "combiner", combiner_preferred = "combiner_preferred", ip = "ip", status = "status", last_seen = start_date, package = "package")
    
        # Add a client and check that we get the added client back
        success, read_client1 = db.client_store.add(c)
        assert success == True
        assert isinstance(read_client1.client_id, str)
        read_client1_dict = read_client1.to_dict()
        client_id = read_client1_dict["client_id"]
        del read_client1_dict["client_id"]
        assert read_client1_dict == c.to_dict()

        # Assert we get the same client back
        read_client2 = db.client_store.get(client_id)
        assert read_client2 is not None
        assert read_client2.to_dict() == read_client1.to_dict()
        
        # Update the client and check that we get the updated client back
        read_client2.name = "new_name"         
        success, read_client3 = db.client_store.update(read_client2)
        assert success == True
        assert read_client3.name == "new_name"

        # Assert we get the same client back
        read_client4 = db.client_store.get(client_id)
        assert read_client4 is not None
        assert read_client3.to_dict() == read_client4.to_dict()

        # Partial update the client and check that we get the updated client back
        update_client = Client(client_id=client_id, combiner="new_combiner")
        success, read_client5 = db.client_store.update(update_client)
        assert success == True
        assert read_client5.combiner == "new_combiner"

        # Assert we get the same client back
        read_client6 = db.client_store.get(client_id)
        assert read_client6 is not None
        assert read_client6.to_dict() == read_client5.to_dict()

        # Delete the client and check that it is deleted
        success = db.client_store.delete(client_id)
        assert success == True

    def test_list_count(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:                        
                gathered_clients = db_1.client_store.select(*opt, **kwargs)
                gathered_clients2 = db_2.client_store.select(*opt, **kwargs)
                count = db_1.client_store.count(**kwargs)
                count2 = db_2.client_store.count(**kwargs)

                assert count == count2
                assert len(gathered_clients) == len(gathered_clients2)

                for i in range(len(gathered_clients)):
                    assert gathered_clients[i].to_dict() == gathered_clients2[i].to_dict()
                





    
    
    
    
    
