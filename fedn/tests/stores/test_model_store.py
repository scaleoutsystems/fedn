import datetime
import itertools
from typing import List
import uuid

import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import ModelDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester


@pytest.fixture
def test_models():
    return [ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name1"),
            ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name2"),
            ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name3"),
            ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name4"),
            ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name5"),
            ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model2", session_id=None, name="test_name6"),
            ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model2", session_id=None, name="test_name7"),
            ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model2", session_id=None, name="test_name8")]


@pytest.fixture
def test_model():
    return ModelDTO(name="model_name", parent_model=None, session_id=None)

@pytest.fixture
def options():
    sorting_keys = (None, 
                    "name",
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"parent_model":"test_parent_model2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestModelStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.model_store
    
    @pytest.fixture
    def test_dto(self, test_model: ModelDTO):
        yield test_model
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_models: List[ModelDTO]):
        yield from self.helper_prepare_and_cleanup(all_db_connections, "model_store", test_models)

    def update_function(self, dto):
        dto.name = "new_name"         
        return dto
    
    def test_add_delete(self, store, test_dto):
        added_dto = store.add(test_dto)
        added_primary_id = added_dto.primary_id
        assert added_dto is not test_dto, "Added DTO should be a new instance, not the original test DTO."
        assert isinstance(added_dto.primary_id, str)
        assert isinstance(added_dto.committed_at, datetime.datetime)
        assert isinstance(added_dto.updated_at, datetime.datetime)

        added_dto_dict = added_dto.to_dict()
        
        del added_dto_dict[added_dto.primary_key]
        del added_dto_dict["model"]
        del added_dto_dict["committed_at"]
        del added_dto_dict["updated_at"]

        test_dto_dict = test_dto.to_dict()
        del test_dto_dict[added_dto.primary_key]
        del test_dto_dict["model"]
        del test_dto_dict["committed_at"]
        del test_dto_dict["updated_at"]

        assert added_dto_dict == test_dto_dict 

        # Delete the attribute and check that it is deleted
        success = store.delete(added_primary_id)
        assert success == True
        
