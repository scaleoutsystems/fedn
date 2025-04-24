import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.validation import ValidationDTO
from fedn.network.storage.statestore.stores.shared import SortOrder

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
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_validations):
    for c in test_validations:
        mongo_connection.validation_store.add(c)
        postgres_connection.validation_store.add(c)
        sql_connection.validation_store.add(c)
    
    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    
    for c in test_validations:
        mongo_connection.validation_store.delete(c.validation_id)
        postgres_connection.validation_store.delete(c.validation_id)
        sql_connection.validation_store.delete(c.validation_id)



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

class TestValidationStore:

    def test_add_update_delete(self, db_connection:DatabaseConnection, test_validation:ValidationDTO):
        test_validation.check_validity()

        read_validation1 = db_connection.validation_store.add(test_validation)
        assert isinstance(read_validation1.validation_id, str)
        assert isinstance(read_validation1.committed_at, datetime.datetime)

        read_validation1_dict = read_validation1.to_dict()
        validation_id = read_validation1_dict["validation_id"]
        del read_validation1_dict["validation_id"]
        del read_validation1_dict["committed_at"]

        test_validation_dict = test_validation.to_dict()
        del test_validation_dict["validation_id"]
        del test_validation_dict["committed_at"]

        assert read_validation1_dict == test_validation_dict

        # Assert we get the same validation back
        read_validation2 = db_connection.validation_store.get(validation_id)
        assert read_validation2 is not None
        assert read_validation2.to_dict() == read_validation1.to_dict()    

        # Delete the validation and check that it is deleted
        success = db_connection.validation_store.delete(validation_id)
        assert success == True

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_validations = db_1.validation_store.list(*opt, **kwargs)
                count = db_1.validation_store.count(**kwargs)
                
                gathered_validations2 = db_2.validation_store.list(*opt, **kwargs)
                count2 = db_2.validation_store.count(**kwargs)
                
                assert count == count2
                assert len(gathered_validations) == len(gathered_validations2)

                for i in range(len(gathered_validations)):
                    assert gathered_validations[i].validation_id == gathered_validations2[i].validation_id



