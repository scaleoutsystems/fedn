import datetime
import itertools
import uuid

import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import ModelDTO
from fedn.network.storage.statestore.stores.shared import SortOrder


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
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_models):
    for c in test_models:
        mongo_connection.model_store.add(c)
        postgres_connection.model_store.add(c)
        sql_connection.model_store.add(c)
        

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for m in test_models:
        mongo_connection.model_store.delete(m.model_id)
        postgres_connection.model_store.delete(m.model_id)
        sql_connection.model_store.delete(m.model_id)
        



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

class TestModelStore:

    def test_add_update_delete(self, db_connection:DatabaseConnection, test_model:ModelDTO):
        test_model.check_validity()

        # Add a model and check that we get the added model back
        read_model1 = db_connection.model_store.add(test_model)
        assert isinstance(read_model1.model_id, str)
        read_model1_dict = read_model1.to_dict()
        model_id = read_model1.model_id
        del read_model1_dict["model_id"]
        del read_model1_dict["model"]
        del read_model1_dict["committed_at"]
        del read_model1_dict["updated_at"]


        input_dict = test_model.to_dict()
        del input_dict["model_id"]
        del input_dict["model"]
        del input_dict["committed_at"]
        del input_dict["updated_at"]

        assert read_model1_dict == input_dict

        # Assert we get the same model back
        read_model2 = db_connection.model_store.get(model_id)
        assert read_model2 is not None
        assert read_model2.to_dict() == read_model1.to_dict()
        
        # Update the model and check that we get the updated model back
        read_model2.name = "new_name"         
        read_model3 = db_connection.model_store.update(read_model2)
        assert read_model3.name == "new_name"

        # Assert we get the same model back
        read_model4 = db_connection.model_store.get(model_id)
        assert read_model4 is not None
        assert read_model3.to_dict() == read_model4.to_dict()


        # Delete the model and check that it is deleted
        success = db_connection.model_store.delete(model_id)
        assert success == True

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options: 
                gathered_models1 = db_1.model_store.list(*opt, **kwargs)
                count1 = db_1.model_store.count(**kwargs)

                gathered_models2 = db_2.model_store.list(*opt, **kwargs)
                count2 = db_2.model_store.count(**kwargs)

                assert count1 == count2
                assert len(gathered_models1) == len(gathered_models2)

                for i in range(len(gathered_models1)):
                    assert gathered_models1[i].model_id == gathered_models2[i].model_id
                    
