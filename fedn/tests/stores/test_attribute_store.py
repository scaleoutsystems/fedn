from typing import List, Tuple
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.attribute import AttributeDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

from fedn.tests.stores.test_store import StoreTester

@pytest.fixture
def test_attributes():    
    timestamp = datetime.datetime.now().replace(microsecond=0)
    attribute1 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data", value = "3", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    attribute2 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data2", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    attribute3 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data2", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    attribute4 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data2", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    
    return [attribute1, attribute2, attribute3, attribute4]

@pytest.fixture
def test_attribute():
    timestamp = datetime.datetime.now().replace(microsecond=0)
    return AttributeDTO(key="data", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1"}, timestamp=timestamp)    

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
    opt_kwargs = ({}, {"sender.name":"test_sender_name1"}, {"key":"data"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))



class TestAttributeStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.attribute_store
    
    @pytest.fixture
    def test_dto(self, test_attribute: AttributeDTO):
        yield test_attribute
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_attributes: List[AttributeDTO]):
        yield from self.helper_prepare_and_clenup(all_db_connections, "attribute_store", test_attributes)
    
    def test_update(self, store, test_dto):
       with pytest.raises(NotImplementedError):
            store.update(test_dto)
