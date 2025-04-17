import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.prediction import PredictionDTO
from fedn.network.storage.statestore.stores.shared import SortOrder




@pytest.fixture
def test_predictions():
    pred1 = PredictionDTO(prediction_id=str(uuid.uuid4()), correlation_id="test_correlation_id1", data="test_data1", model_id=None,
                          receiver={"name":"test_receiver_name1", "role":"test_receiver_role1"}, sender={"name":"test_sender_name1",
                          "role":"test_sender_role1"}, timestamp="2024-13")
    pred2 = PredictionDTO(prediction_id=str(uuid.uuid4()), correlation_id="test_correlation_id2", data="test_data1", model_id=None,
                            receiver={"name":"test_receiver_name2", "role":"test_receiver_role1"}, sender={"name":"test_sender_name1",
                            "role":"test_sender_role1"}, timestamp="2024-11")
    pred3 = PredictionDTO(prediction_id=str(uuid.uuid4()), correlation_id="test_correlation_id3", data="test_data1", model_id=None,
                            receiver={"name":"test_receiver_name3", "role":"test_receiver_role1"}, sender={"name":"test_sender_name2",
                            "role":"test_sender_role1"}, timestamp="2024-12")
    pred4 = PredictionDTO(prediction_id=str(uuid.uuid4()), correlation_id="test_correlation_id4", data="test_data1", model_id=None,
                            receiver={"name":"test_receiver_name4", "role":"test_receiver_role1"}, sender={"name":"test_sender_name2",
                            "role":"test_sender_role1"}, timestamp="2024-16")
    return [pred1, pred2, pred3, pred4]



@pytest.fixture
def test_prediction():
    prediction = PredictionDTO()
    prediction.correlation_id = "test_correlation_id1"
    prediction.data = "test_data1"
    prediction.model_id = None
    prediction.receiver.name = "test_receiver_name1"
    prediction.receiver.role = "test_receiver_role1"
    prediction.sender.name = "test_sender_name1"
    prediction.sender.role = "test_sender_role1"
    prediction.timestamp = "2024-13"
    return prediction


@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_predictions):
    for c in test_predictions:
        mongo_connection.prediction_store.add(c)
        postgres_connection.prediction_store.add(c)
        sql_connection.prediction_store.add(c)

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_predictions:
        mongo_connection.prediction_store.delete(c.prediction_id)
        postgres_connection.prediction_store.delete(c.prediction_id)
        sql_connection.prediction_store.delete(c.prediction_id)



@pytest.fixture
def options():
    sorting_keys = (None, 
                    "correlation_id", 
                    "timestamp", 
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"correlation_id":""})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestPredictionStore:

    def test_add_update_delete(self, db_connection: DatabaseConnection, test_prediction: PredictionDTO):
        test_prediction.check_validity()

        read_prediction1 = db_connection.prediction_store.add(test_prediction)
        assert isinstance(read_prediction1.prediction_id, str)
        assert isinstance(read_prediction1.committed_at, datetime.datetime)

        read_prediction1_dict = read_prediction1.to_dict()
        prediction_id = read_prediction1_dict["prediction_id"]
        del read_prediction1_dict["prediction_id"]
        del read_prediction1_dict["committed_at"]

        test_prediction_dict = test_prediction.to_dict()
        del test_prediction_dict["prediction_id"]
        del test_prediction_dict["committed_at"]

        assert read_prediction1_dict == test_prediction_dict

        # Assert we get the same prediction back
        read_prediction2 = db_connection.prediction_store.get(prediction_id)
        assert read_prediction2 is not None
        assert read_prediction2.to_dict() == read_prediction1.to_dict()    

        # Delete the prediction and check that it is deleted
        success = db_connection.prediction_store.delete(prediction_id)
        assert success == True
    

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_models = db_1.prediction_store.list(*opt, **kwargs)
                count = db_1.prediction_store.count(**kwargs)

                gathered_models2 = db_2.prediction_store.list(*opt, **kwargs)
                count2 = db_2.prediction_store.count(**kwargs)

                assert count == count2
                assert len(gathered_models) == len(gathered_models2)

                for i in range(len(gathered_models)):
                    assert gathered_models[i].prediction_id == gathered_models2[i].prediction_id
                    
