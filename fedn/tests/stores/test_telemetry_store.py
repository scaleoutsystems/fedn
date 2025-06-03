from typing import List, Tuple
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import TelemetryDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

from fedn.tests.stores.test_store import StoreTester

@pytest.fixture
def test_telemetries():    
    timestamp = datetime.datetime.now().replace(microsecond=0)
    telemetry1 = TelemetryDTO(telemetry_id = str(uuid.uuid4()), key="data", value = 3, sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    telemetry2 = TelemetryDTO(telemetry_id = str(uuid.uuid4()), key="data", value = 4, sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    telemetry3 = TelemetryDTO(telemetry_id = str(uuid.uuid4()), key="data2", value = 3, sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    telemetry4 = TelemetryDTO(telemetry_id = str(uuid.uuid4()), key="data2", value = 6, sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    return [telemetry1, telemetry2, telemetry3, telemetry4]

@pytest.fixture
def test_telemetry():
    timestamp = datetime.datetime.now().replace(microsecond=0)
    return TelemetryDTO(key="data2", value = 6, sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)


@pytest.fixture
def options():
    sorting_keys = (None, 
                    "sender.name", 
                    "timestamp", 
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"sender.name":"test_sender_name1"}, {"key":"data2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))



class TestTelemetryStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.telemetry_store
    
    @pytest.fixture
    def test_dto(self, test_telemetry: TelemetryDTO):
        yield test_telemetry
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_telemetries: List[TelemetryDTO]):
        yield from self.helper_prepare_and_clenup(all_db_connections, "telemetry_store", test_telemetries)
    
    def test_update(self, store, test_dto):
       with pytest.raises(NotImplementedError):
            store.update(test_dto)
