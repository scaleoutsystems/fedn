from typing import List
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.validation import ValidationDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester

@pytest.fixture
def test_validations():
    start_time = datetime.datetime(2021, 1, 4, 1, 2, 4)
   
    validation1 = ValidationDTO(validation_id=str(uuid.uuid4()), data="test_data1", timestamp=start_time+datetime.timedelta(days=2), correlation_id=str(uuid.uuid4()))
    validation2 = ValidationDTO(validation_id=str(uuid.uuid4()), data="test_data2", timestamp=start_time+datetime.timedelta(days=4), correlation_id=str(uuid.uuid4()))
    validation3 = ValidationDTO(validation_id=str(uuid.uuid4()), data="test_data3", timestamp=start_time+datetime.timedelta(days=5), correlation_id=str(uuid.uuid4()))
    validation4 = ValidationDTO(validation_id=str(uuid.uuid4()), data="test_data4", timestamp=start_time+datetime.timedelta(days=1), correlation_id=str(uuid.uuid4()))
    validation5 = ValidationDTO(validation_id=str(uuid.uuid4()), data="test_data5", timestamp=start_time+datetime.timedelta(days=6), correlation_id=str(uuid.uuid4()))

    return [validation1, validation2, validation3, validation4, validation5]

 

@pytest.fixture
def test_validation():
    start_time = datetime.datetime(2021, 1, 4, 1, 2, 4)
    return ValidationDTO(data="test_data1", timestamp=start_time, correlation_id=str(uuid.uuid4()))

@pytest.fixture
def options():
    sorting_keys = (None, 
                    "data",
                    "timestamp",
                    "invalid_key"
                        ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"data":"test_data6"}, {"data":""})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestValidationStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.validation_store
    
    @pytest.fixture
    def test_dto(self, test_validation: ValidationDTO):
        yield test_validation
            
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_validations: List[ValidationDTO]):
        yield from self.helper_prepare_and_cleanup(all_db_connections, "validation_store", test_validations)

    def test_update(self, store, test_dto):
        with pytest.raises(NotImplementedError):
            store.update(test_dto)

