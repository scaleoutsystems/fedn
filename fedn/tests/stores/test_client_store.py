import datetime
import itertools
from typing import List
import uuid

import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import ClientDTO
from fedn.network.storage.statestore.stores.shared import SortOrder


@pytest.fixture
def test_clients():
    #TODO: SQL versions does not support updated_at field
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    return [
        ClientDTO(client_id=str(uuid.uuid4()), name="test_client1", combiner="test_combiner", ip="121.12.32.2", combiner_preferred="", status="test_status", last_seen=start_date - datetime.timedelta(days=1), package="local"),
        ClientDTO(client_id=str(uuid.uuid4()), name="test_client2", combiner="test_combiner", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(days=3), package="local"),
        ClientDTO(client_id=str(uuid.uuid4()), name="test_client3", combiner="test_combiner", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(days=4), package="local"),
        ClientDTO(client_id=str(uuid.uuid4()), name="test_client5", combiner="test_combiner", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(seconds=5), package="remote"),
        ClientDTO(client_id=str(uuid.uuid4()), name="test_client4", combiner="test_combiner2", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(weeks=1), package="remote"),
        ClientDTO(client_id=str(uuid.uuid4()), name="test_client6", combiner="test_combiner2", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(hours=1), package="local"),
        ClientDTO(client_id=str(uuid.uuid4()), name="test_client7", combiner="test_combiner2", ip="121.12.32.22", combiner_preferred="", status="test_status2", last_seen=start_date - datetime.timedelta(minutes=1), package="remote"),
    ]

@pytest.fixture
def test_client():
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    c = ClientDTO(name = "name", combiner = "combiner", combiner_preferred = "combiner_preferred", ip = "ip", status = "status", last_seen = start_date, package = "package")
    return c

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_clients: List[ClientDTO]):
    for c in test_clients:
        mongo_connection.client_store.add(c)
        postgres_connection.client_store.add(c)
        sql_connection.client_store.add(c)

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_clients:
        mongo_connection.client_store.delete(c.client_id)
        postgres_connection.client_store.delete(c.client_id)
        sql_connection.client_store.delete(c.client_id)



@pytest.fixture
def options():
    sorting_keys = (None,
                    "name",
                    "client_id",
                    "last_seen",
                    "ip", # None unique key 
                    "invalid_key",
                    "committed_at"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"ip":"121.12.32.22"}, {"combiner":"test_combiner", "status":"test_status"}, {"name":"test_client1", "status":"test_status2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestClientStore:

    def test_add_get_update_delete(self, db_connection:DatabaseConnection, test_client: ClientDTO): 
        test_client.check_validity()

        # Add a client and check that we get the added client back
        read_client1 = db_connection.client_store.add(test_client)
        assert isinstance(read_client1.client_id, str)
        read_client1_dict = read_client1.to_dict()
        client_id = read_client1_dict["client_id"]
        del read_client1_dict["client_id"]
        del read_client1_dict["committed_at"]
        del read_client1_dict["updated_at"]

        test_client_dict = test_client.to_dict()
        del test_client_dict["client_id"]
        del test_client_dict["committed_at"]
        del test_client_dict["updated_at"]

        assert read_client1_dict == test_client_dict

        # Assert we get the same client back
        read_client2 = db_connection.client_store.get(client_id)
        assert read_client2 is not None
        assert read_client2.to_dict() == read_client1.to_dict()
        
        # Update the client and check that we get the updated client back
        read_client2.name = "new_name"         
        read_client3 = db_connection.client_store.update(read_client2)
        assert read_client3.name == "new_name"

        # Assert we get the same client back
        read_client4 = db_connection.client_store.get(client_id)
        assert read_client4 is not None
        assert read_client3.to_dict() == read_client4.to_dict()

        # Delete the client and check that it is deleted
        success = db_connection.client_store.delete(client_id)
        assert success == True

    def test_list_count(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:                        
                gathered_clients = db_1.client_store.list(*opt, **kwargs)
                gathered_clients2 = db_2.client_store.list(*opt, **kwargs)
                count = db_1.client_store.count(**kwargs)
                count2 = db_2.client_store.count(**kwargs)

                assert count == count2
                assert len(gathered_clients) == len(gathered_clients2)

                for i in range(len(gathered_clients)):
                    assert gathered_clients[i].client_id == gathered_clients2[i].client_id
                





    
    
    
    
    
