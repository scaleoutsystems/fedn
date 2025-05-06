import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.status import StatusDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

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
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_statuses):
    for c in test_statuses:
        mongo_connection.status_store.add(c)
        postgres_connection.status_store.add(c)
        sql_connection.status_store.add(c)
        
    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    
    for c in test_statuses:
        mongo_connection.status_store.delete(c.status_id)
        postgres_connection.status_store.delete(c.status_id)
        sql_connection.status_store.delete(c.status_id)


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




class TestStatusStore:

    def test_add_update_delete(self, db_connection: DatabaseConnection, test_status: StatusDTO):
        test_status.check_validity()

        read_status1 = db_connection.status_store.add(test_status)
        assert isinstance(read_status1.status_id, str)
        assert isinstance(read_status1.committed_at, datetime.datetime)

        read_status1_dict = read_status1.to_dict()
        status_id = read_status1_dict["status_id"]
        del read_status1_dict["status_id"]
        del read_status1_dict["committed_at"]
        del read_status1_dict["updated_at"]

        test_status_dict = test_status.to_dict()
        del test_status_dict["status_id"]
        del test_status_dict["committed_at"]
        del test_status_dict["updated_at"]

        assert read_status1_dict == test_status_dict

        # Assert we get the same status back
        read_status2 = db_connection.status_store.get(status_id)
        assert read_status2 is not None
        assert read_status2.to_dict() == read_status1.to_dict()    

        # Delete the status and check that it is deleted
        success = db_connection.status_store.delete(status_id)
        assert success == True

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_statuses = db_1.status_store.list(*opt, **kwargs)
                count =  db_1.status_store.count(**kwargs)

                gathered_statuses2 = db_2.status_store.list(*opt, **kwargs)
                count2 =  db_2.status_store.count(**kwargs)

                assert count == count2
                assert len(gathered_statuses) == len(gathered_statuses2)

                for i in range(len(gathered_statuses)):
                    assert gathered_statuses[i].status_id == gathered_statuses2[i].status_id


