from typing import List
import pytest


import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import CombinerDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

from fedn.tests.stores.test_store import StoreTester


@pytest.fixture
def test_combiners():
    combiner1 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner1",
                  parent="localhost", ip="123:13:12:2", fqdn="", port=8080, address="test_address")
    combiner2 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner2",
                  parent="localhost", ip="123:13:12:2", fqdn="", port=8080, address="test_address") 
    combiner3 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner3",
                    parent="localhost", ip="123:13:12:5", fqdn="", port=8080, address="test_address")
    combiner4 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner4",
                    parent="localhost", ip="123:13:12:4", fqdn="", port=8080, address="test_address")
    combiner5 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner5",
                    parent="localhost", ip="123:13:12:3", fqdn="", port=8080, address="test_address")
    combiner6 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner6",
                    parent="localhost", ip="123:13:12:3", fqdn="", port=8080, address="test_address")
    combiner7 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner8",
                    parent="localhost", ip="123:13:12:3", fqdn="", port=8080, address="test_address")
    combiner8 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner7",
                    parent="localhost", ip="123:13:12:2", fqdn="", port=8080, address="test_address1")
    return [combiner1, combiner2, combiner3, combiner4, combiner5, combiner6, combiner7, combiner8]

@pytest.fixture
def test_combiner():
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    combiner = CombinerDTO(name="test_combiner",
                  parent="localhost", ip="123:13:12:2", fqdn="", port=8080,
                  updated_at=start_date - datetime.timedelta(days=52), address="test_address")
    return combiner    

@pytest.fixture
def options():
    sorting_keys = (None, 
                    "name",
                    "committed_at",
                    "updated_at",
                    "invalid_key") 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"ip":"123:13:12:3"}, {"address":"test_address1"}, {"fqdn":"", "name":"test_combiner77"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))



class TestCombinerStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.combiner_store
    
    @pytest.fixture
    def test_dto(self, test_combiner: CombinerDTO):
        yield test_combiner
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_combiners: List[CombinerDTO]):
        yield from self.helper_prepare_and_clenup(all_db_connections, "combiner_store", test_combiners)

    def test_update(self, store, test_dto):
       with pytest.raises(NotImplementedError):
            store.update(test_dto)