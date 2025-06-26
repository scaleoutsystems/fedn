import datetime
import itertools
from typing import List
import uuid

import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import RunDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

from fedn.tests.stores.test_store import  StoreTester

@pytest.fixture
def test_runs():
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    runs =  [
        RunDTO(run_id=str(uuid.uuid4()), round_timeout=60, rounds=10, completed_at=start_date - datetime.timedelta(days=1)),
        RunDTO(run_id=str(uuid.uuid4()), round_timeout=60, rounds=10, completed_at=start_date - datetime.timedelta(days=3)),
        RunDTO(run_id=str(uuid.uuid4()), round_timeout=60, rounds=5, completed_at=start_date - datetime.timedelta(days=2)),
        RunDTO(run_id=str(uuid.uuid4()), round_timeout=60, rounds=5, completed_at=start_date - datetime.timedelta(seconds=2)),
    ]
    return runs

@pytest.fixture
def test_run():
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    run = RunDTO(run_id=str(uuid.uuid4()), round_timeout=60, rounds=10, completed_at=start_date - datetime.timedelta(days=1))
    return run

@pytest.fixture
def options():
    sorting_keys = (None,
                    "round_timeout",
                    "run_id",
                    "committed_at",
                    "invalid_key",
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"rounds":"5"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))



class TestRunStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.run_store
    
    @pytest.fixture
    def test_dto(self, test_run: RunDTO):
        yield test_run
            
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_runs: List[RunDTO]):
        yield from self.helper_prepare_and_cleanup(all_db_connections, "run_store", test_runs)

    def update_function(self, dto):
        dto.rounds = 1
        return dto

                





    
    
    
    
    
