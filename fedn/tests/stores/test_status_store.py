from typing import List
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.status import StatusDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester

@pytest.fixture
def test_statuses():
    start_time = datetime.datetime(2021, 1, 4, 1, 2, 4)
   
    status1 = StatusDTO(status_id=str(uuid.uuid4()), status="status1", timestamp=start_time+datetime.timedelta(days=2), log_level="test_log_level1", type="test_type", sender={"name":"test_sender", "role":"test_role"})
    status2 = StatusDTO(status_id=str(uuid.uuid4()), status="status2", timestamp=start_time+datetime.timedelta(days=4), log_level="test_log_level2", type="test_type", sender={"name":"test_sender", "role":"test_role"})
    status3 = StatusDTO(status_id=str(uuid.uuid4()), status="status3", timestamp=start_time+datetime.timedelta(days=5), log_level="test_log_level3", type="test_type", sender={"name":"test_sender", "role":"test_role"})
    status4 = StatusDTO(status_id=str(uuid.uuid4()), status="status4", timestamp=start_time+datetime.timedelta(days=1), log_level="test_log_level4", type="test_type2", sender={"name":"test_sender", "role":"test_role"})
    status5 = StatusDTO(status_id=str(uuid.uuid4()), status="status5", timestamp=start_time+datetime.timedelta(days=6), log_level="test_log_level5", type="test_type2", sender={"name":"test_sender", "role":"test_role"})

    return [status1, status2, status3, status4, status5]

@pytest.fixture
def test_status():
    start_time = datetime.datetime(2021, 1, 4, 1, 2, 4)
    return StatusDTO(status_id=str(uuid.uuid4()), status="status1", timestamp=start_time, log_level="test_log_level1", type="test_type", sender={"name":"test_sender", "role":"test_role"})

@pytest.fixture
def options():
    sorting_keys = (None, 
                    "log_level",
                    "timestamp",
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"log_level":"test_log_level6"}, {"type":"test_type2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))



class TestStatusStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.status_store
    
    @pytest.fixture
    def test_dto(self, test_status: StatusDTO):
        yield test_status
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_statuses: List[StatusDTO]):
        yield from self.helper_prepare_and_cleanup(all_db_connections, "status_store", test_statuses)

    def test_update(self, store, test_dto):
        with pytest.raises(NotImplementedError):
            store.update(test_dto)


