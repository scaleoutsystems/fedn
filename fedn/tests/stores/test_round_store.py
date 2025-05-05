from typing import Tuple
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.model import ModelDTO
from fedn.network.storage.statestore.stores.dto.round import RoundDTO
from fedn.network.storage.statestore.stores.shared import SortOrder



@pytest.fixture
def test_rounds():

    return [RoundDTO(round_id=str(uuid.uuid4()), status="test_status6"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status4"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status3"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status1"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status2"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status5")]

@pytest.fixture
def test_round_model():
    model = ModelDTO(model_id=str(uuid.uuid4()), name="test_model")

    round = RoundDTO(round_id=str(uuid.uuid4()), status="test_status6")
    round.round_config = {"aggregator":"test_aggregator", "buffer_size":100, "delete_models_storage":True, "clients_required":10,
                          "requested_clients":10,"helper_type":"test_helper_type", "model_id":model.model_id, "round_timeout":100,
                          "validate":True, "round_id":round.round_id, "task":"test_task", "rounds":1, "client_settings":{"test_key":"test_value"}}
    
    round.round_data = {"time_commit":100}
    round.round_data.reduce = {"time_aggregate_model":100, "time_fetch_model":100, "time_load_model":100}
    round.combiners = [{"model_id":model.model_id, "name":"test_name1", "status":"test_status1", "time_exec_training":100,
                              "config":round.round_config.to_dict(),
                              "data":{"aggregation_time":{"nr_aggregated_models":10, "time_model_aggregation":100, "time_model_load":100},
                                      "nr_expected_updates":10, "nr_required_updates":10, "time_combination":100, "timeout":100}}]
    return round, model


@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_rounds):
    for c in test_rounds:
        mongo_connection.round_store.add(c)
        postgres_connection.round_store.add(c)
        sql_connection.round_store.add(c)

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_rounds:
        mongo_connection.round_store.delete(c.round_id)
        postgres_connection.round_store.delete(c.round_id)
        sql_connection.round_store.delete(c.round_id)


    



@pytest.fixture
def options():
    sorting_keys = (None, 
                    "status", 
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"status":"test_status4,test_status5"}, {"status":"blah"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestRoundStore:

    def test_add_update_delete(self, db_connection: DatabaseConnection, test_round_model: Tuple[RoundDTO, ModelDTO]):
        test_round, test_model = test_round_model

        db_connection.model_store.add(test_model)

        test_round.check_validity()

        # Add a round and check that we get the added round back
        read_round1 = db_connection.round_store.add(test_round)
        assert isinstance(read_round1.round_id, str)
        read_round1_dict = read_round1.to_dict()
        round_id = read_round1_dict["round_id"]
        del read_round1_dict["round_id"]
        del read_round1_dict["committed_at"]
        del read_round1_dict["updated_at"]

        test_round_dict = test_round.to_dict()
        del test_round_dict["round_id"]
        del test_round_dict["committed_at"]
        del test_round_dict["updated_at"]

        assert read_round1_dict == test_round_dict

        # Assert we get the same round back
        read_round2 = db_connection.round_store.get(round_id)
        assert read_round2 is not None
        assert read_round2.to_dict() == read_round1.to_dict()
        
        # Update the round and check that we get the updated round back
        read_round2.round_config.aggregator = "new_aggregator"         
        read_round3 = db_connection.round_store.update(read_round2)
        assert read_round3.round_config.aggregator == "new_aggregator"

        # Assert we get the same round back
        read_round4 = db_connection.round_store.get(round_id)
        assert read_round4 is not None
        assert read_round3.to_dict() == read_round4.to_dict()

        # Add combiner to round and check that we get the updated round back
        combiner = read_round4.combiners[0]
        combiner.status = "new_status"
        read_round4.combiners.append(combiner)
        read_round5 = db_connection.round_store.update(read_round4)
        assert len(read_round5.combiners) == 2
        assert read_round5.combiners[1].status == "new_status"

        # Assert we get the same round back
        read_round6 = db_connection.round_store.get(round_id)
        assert read_round6 is not None
        assert read_round5.to_dict() == read_round6.to_dict()


        # Remove config from round and check that we get the updated round back
        read_round6.round_config = None
        read_round7 = db_connection.round_store.update(read_round6)
        assert read_round7.to_dict()["round_config"] == None

        # Assert we get the same round back
        read_round8 = db_connection.round_store.get(round_id)
        assert read_round8 is not None
        assert read_round7.to_dict() == read_round8.to_dict()

        #Remove all combiners from round and check that we get the updated round back
        read_round8.combiners = []
        read_round9 = db_connection.round_store.update(read_round8)
        assert len(read_round9.combiners) == 0

        # Assert we get the same round back
        read_round10 = db_connection.round_store.get(round_id)
        assert read_round10 is not None
        assert read_round9.to_dict() == read_round10.to_dict()
        

        # Delete the round and check that it is deleted
        success = db_connection.round_store.delete(round_id)
        assert success == True
        
        db_connection.model_store.delete(test_model.model_id)


    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_rounds = db_1.round_store.list(*opt, **kwargs)
                count = db_1.round_store.count(**kwargs)

                gathered_rounds2 = db_2.round_store.list(*opt, **kwargs)
                count2 = db_2.round_store.count(**kwargs)
                
                assert count == count2
                assert len(gathered_rounds) == len(gathered_rounds2)

                for i in range(len(gathered_rounds)):
                    assert gathered_rounds2[i].round_id == gathered_rounds[i].round_id


