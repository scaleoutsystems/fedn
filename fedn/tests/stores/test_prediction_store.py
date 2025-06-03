from typing import List
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.prediction import PredictionDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester




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

class TestPredictionStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.prediction_store
    
    @pytest.fixture
    def test_dto(self, test_prediction: PredictionDTO):
        yield test_prediction
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_predictions: List[PredictionDTO]):
        yield from self.helper_prepare_and_clenup(all_db_connections, "prediction_store", test_predictions)

    def test_update(self, store, test_dto):
       with pytest.raises(NotImplementedError):
            store.update(test_dto)
