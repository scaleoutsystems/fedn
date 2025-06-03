from typing import List, Tuple
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.metric import MetricDTO
from fedn.network.storage.statestore.stores.dto.model import ModelDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester



@pytest.fixture
def test_model_metrics():
    model = ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name1")
    
    timestamp = datetime.datetime.now().replace(microsecond=0)
    

    metric1 = MetricDTO(metric_id=str(uuid.uuid4()),key="loss", value=0.1, sender={"name":"test_sender_name2", "role":"test_sender_role1"}, timestamp=timestamp, model_id=model.model_id)
    metric2 = MetricDTO(metric_id=str(uuid.uuid4()),key="loss", value=0.1, sender={"name":"test_sender_name1", "role":"test_sender_role1"}, timestamp=timestamp, model_id=model.model_id)
    metric3 = MetricDTO(metric_id=str(uuid.uuid4()),key="accuracy", value=0.1, sender={"name":"test_sender_name2", "role":"test_sender_role1"}, timestamp=timestamp, model_id=model.model_id)
    metric4 = MetricDTO(metric_id=str(uuid.uuid4()),key="accuracy", value=0.1, sender={"name":"test_sender_name1", "role":"test_sender_role1"}, timestamp=timestamp, model_id=model.model_id)
    return model, [metric1, metric2, metric3, metric4]



@pytest.fixture
def test_model_metric():
    timestamp = datetime.datetime.now().replace(microsecond=0)

    model = ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name1")
    metric = MetricDTO()
    metric.key = "loss"
    metric.value = 0.1
    metric.sender.name = "test_sender_name1"
    metric.sender.role = "test_sender_role1"
    metric.timestamp = timestamp
    metric.model_id = model.model_id
    metric.step = 0
    return model, metric

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
    opt_kwargs = ({}, {"key":"loss"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestMetricStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.metric_store
    
    @pytest.fixture
    def test_dto(self, db_connection, test_model_metric: MetricDTO):
        model, test_metric = test_model_metric
        db_connection.model_store.add(model)
        yield test_metric
        db_connection.model_store.delete(model.model_id)
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_model_metrics: List[MetricDTO]):
        model, test_metrics = test_model_metrics
        for connection_name, connection in all_db_connections:
            connection.model_store.add(model) 
        yield from self.helper_prepare_and_cleanup(all_db_connections, "metric_store", test_metrics)
        for connection_name, connection in all_db_connections:
            connection.model_store.delete(model.model_id)

    def test_update(self, store, test_dto):
       with pytest.raises(NotImplementedError):
            store.update(test_dto)