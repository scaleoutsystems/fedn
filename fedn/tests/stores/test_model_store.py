import pytest
import pymongo

import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import ModelDTO


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
        res, _ = mongo_connection.model_store.add(c)
        assert res == True

    for c in test_models:
        res, _ = postgres_connection.model_store.add(c)
        assert res == True

    for c in test_models:
        res, _ = sql_connection.model_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for m in test_models:
        res = mongo_connection.model_store.delete(m.model_id)
        assert res == True
    
    for m in test_models:
        res = postgres_connection.model_store.delete(m.model_id)
        assert res == True
    
    for m in test_models:
        res = sql_connection.model_store.delete(m.model_id)
        assert res == True
        



@pytest.fixture
def options():
    sorting_keys = (None, 
                    "name",
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"parent_model":"test_parent_model2"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestModelStore:

    def test_add_update_delete_postgres(self, postgres_connection:DatabaseConnection, test_model):
        self.helper_add_update_delete(postgres_connection, test_model)

    def test_add_update_delete_sqlite(self, sql_connection:DatabaseConnection, test_model):
        self.helper_add_update_delete(sql_connection, test_model)
    
    def test_add_update_delete_mongo(self, mongo_connection:DatabaseConnection, test_model):
        self.helper_add_update_delete(mongo_connection, test_model)

    def helper_add_update_delete(self, db:DatabaseConnection, test_model:ModelDTO):
        # Add a model and check that we get the added model back
        success, read_model1 = db.model_store.add(test_model)
        assert success == True
        assert isinstance(read_model1.model_id, str)
        read_client1_dict = read_model1.to_dict()
        model_id = read_client1_dict["model"]
        del read_client1_dict["model"]
        assert read_client1_dict == test_model.to_dict()

        # Assert we get the same model back
        read_model2 = db.model_store.get(model_id)
        assert read_model2 is not None
        assert read_model2.to_dict() == read_model1.to_dict()
        
        # Update the model and check that we get the updated model back
        read_model2.name = "new_name"         
        success, read_model3 = db.model_store.update(read_model2)
        assert success == True
        assert read_model3.name == "new_name"

        # Assert we get the same model back
        read_model4 = db.model_store.get(model_id)
        assert read_model4 is not None
        assert read_model3.to_dict() == read_model4.to_dict()

        # Partial update the model and check that we get the updated model back
        update_model = ModelDTO(model_id=model_id, parent_model="new_parent")            
        success, read_model5 = db.model_store.update(update_model)
        assert success == True
        assert read_model5.parent_model == "new_parent"

        # Assert we get the same model back
        read_model6 = db.model_store.get(model_id)
        assert read_model6 is not None
        assert read_model6.to_dict() == read_model5.to_dict()

        # Delete the model and check that it is deleted
        success = db.model_store.delete(model_id)
        assert success == True

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options: 
                gathered_models1 = db_1.model_store.select(*opt, **kwargs)
                count1 = db_1.model_store.count(**kwargs)

                gathered_models2 = db_2.model_store.select(*opt, **kwargs)
                count2 = db_2.model_store.count(**kwargs)

                assert count1 == count2
                assert len(gathered_models1) == len(gathered_models2)

                for i in range(len(gathered_models1)):
                    assert gathered_models1[i].to_dict() == gathered_models2[i].to_dict()
                    
                    
