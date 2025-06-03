import datetime
import itertools
from typing import List
import uuid

import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import ClientDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

from fedn.tests.stores.test_store import  StoreTester


@pytest.fixture
def test_clients():
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



class TestClientStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.client_store
    
    @pytest.fixture
    def test_dto(self, test_client: ClientDTO):
        yield test_client
            
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_clients: List[ClientDTO]):
        yield from self.helper_prepare_and_clenup(all_db_connections, "client_store", test_clients)

    def update_function(self, dto):
        dto.name = "updated_name"
        return dto

                





    
    
    
    
    
