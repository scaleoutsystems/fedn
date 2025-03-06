import pytest
import pymongo

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.prediction import PredictionDTO




@pytest.fixture
def test_predictions():
    return [{"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id1", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name1", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name1", 
             "sender_role":"test_sender_role1", "timestamp":"2024-13", "prediction_id":"test_prediction_id1"},
             {"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id4", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name2", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name1", 
             "sender_role":"test_sender_role1", "timestamp":"2024-11", "prediction_id":"test_prediction_id1"},
             {"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id3", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name3", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name2", 
             "sender_role":"test_sender_role1", "timestamp":"2024-12", "prediction_id":"test_prediction_id2"},
             {"id":str(uuid.uuid4()), "correlation_id":"test_correlation_id2", "data":"test_data1", "model_id":None, 
             "receiver_name":"test_receiver_name4", "receiver_role":"test_receiver_role1", "sender_name":"test_sender_name2", 
             "sender_role":"test_sender_role1", "timestamp":"2024-16", "prediction_id":"test_prediction_id2"}]

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_predictions):
    for c in test_predictions:
        res, _ = mongo_connection.prediction_store.add(c)
        assert res == True

    for c in test_predictions:
        res, _ = postgres_connection.prediction_store.add(c)
        assert res == True

    for c in test_predictions:
        res, _ = sql_connection.prediction_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                        "correlation_id", 
                        "timestamp", 
                        #"invalid_key"
                        ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"prediction_id":"test_prediction_id2"}, {"correlation_id":""})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestPredictionStore:

    def helper_add_update_delete(self, db: DatabaseConnection, prediction: PredictionDTO):
         success, read_package1 = db.package_store.add(package)
        assert success == True
        assert isinstance(read_package1.id, str)
        assert isinstance(read_package1.committed_at, datetime.datetime)
        assert isinstance(read_package1.storage_file_name, str)

        read_package1_dict = read_package1.to_dict()
        package_id = read_package1_dict["id"]
        del read_package1_dict["id"]
        del read_package1_dict["committed_at"]
        del read_package1_dict["storage_file_name"]

        test_package_dict = package.to_dict()
        del test_package_dict["id"]
        del test_package_dict["committed_at"]
        del test_package_dict["storage_file_name"]

        assert read_package1_dict == test_package_dict

        # Assert we get the same package back
        read_package2 = db.package_store.get(package_id)
        assert read_package2 is not None
        assert read_package2.to_dict() == read_package1.to_dict()    

        # Delete the package and check that it is deleted
        success = db.package_store.delete(package_id)
        assert success == True
    

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_models = db_1.prediction_store.select(*opt, **kwargs)
                count = db_1.prediction_store.count(**kwargs)

                gathered_models2 = db_2.prediction_store.select(*opt, **kwargs)
                count2 = db_2.prediction_store.count(**kwargs)

                assert count == count2
                assert len(gathered_models) == len(gathered_models2)

                for i in range(len(gathered_models)):
                    assert gathered_models[i].prediction_id == gathered_models2[i].prediction_id
                    
