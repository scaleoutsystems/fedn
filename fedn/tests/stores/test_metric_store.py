import pytest
import pymongo

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.metric import MetricDTO




@pytest.fixture
def test_metrics():
    
    metric1 = MetricDTO(metric_id=str(uuid.uuid4()),key="loss", value=0.1, sender={"name":"test_sender_name2", "role":"test_sender_role1"}, timestamp=datetime.datetime.now())
    metric2 = MetricDTO(metric_id=str(uuid.uuid4()),key="loss", value=0.1, sender={"name":"test_sender_name1", "role":"test_sender_role1"}, timestamp=datetime.datetime.now())
    metric3 = MetricDTO(metric_id=str(uuid.uuid4()),key="accuracy", value=0.1, sender={"name":"test_sender_name2", "role":"test_sender_role1"}, timestamp=datetime.datetime.now())
    metric4 = MetricDTO(metric_id=str(uuid.uuid4()),key="accuracy", value=0.1, sender={"name":"test_sender_name1", "role":"test_sender_role1"}, timestamp=datetime.datetime.now())
    return [metric1, metric2, metric3, metric4]



@pytest.fixture
def test_metric():
    metric = MetricDTO()
    metric.key = "loss"
    metric.value = 0.1
    metric.sender.name = "test_sender_name1"
    metric.sender.role = "test_sender_role1"
    metric.timestamp = datetime.datetime.now()
    metric.model_id = 
    metric.step = 1
    return metric


@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_metrics):
    for c in test_metrics:
        mongo_connection.metric_store.add(c)
        postgres_connection.metric_store.add(c)
        sql_connection.metric_store.add(c)

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_metrics:
        mongo_connection.metric_store.delete(c.metric_id)
        postgres_connection.metric_store.delete(c.metric_id)
        sql_connection.metric_store.delete(c.metric_id)



@pytest.fixture
def options():
    sorting_keys = (None, 
                    "correlation_id", 
                    "timestamp", 
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"correlation_id":""})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestmetricStore:

    def test_add_update_delete(self, db_connection: DatabaseConnection, test_metric: MetricDTO):
        test_metric.check_validity()

        read_metric1 = db_connection.metric_store.add(test_metric)
        assert isinstance(read_metric1.metric_id, str)
        assert isinstance(read_metric1.committed_at, datetime.datetime)

        read_metric1_dict = read_metric1.to_dict()
        metric_id = read_metric1_dict["metric_id"]
        del read_metric1_dict["metric_id"]
        del read_metric1_dict["committed_at"]

        test_metric_dict = test_metric.to_dict()
        del test_metric_dict["metric_id"]
        del test_metric_dict["committed_at"]

        assert read_metric1_dict == test_metric_dict

        # Assert we get the same metric back
        read_metric2 = db_connection.metric_store.get(metric_id)
        assert read_metric2 is not None
        assert read_metric2.to_dict() == read_metric1.to_dict()    

        # Delete the metric and check that it is deleted
        success = db_connection.metric_store.delete(metric_id)
        assert success == True
    

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_models = db_1.metric_store.list(*opt, **kwargs)
                count = db_1.metric_store.count(**kwargs)

                gathered_models2 = db_2.metric_store.list(*opt, **kwargs)
                count2 = db_2.metric_store.count(**kwargs)

                assert count == count2
                assert len(gathered_models) == len(gathered_models2)

                for i in range(len(gathered_models)):
                    assert gathered_models[i].metric_id == gathered_models2[i].metric_id
                    
