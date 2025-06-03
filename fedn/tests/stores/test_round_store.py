from typing import List, Tuple
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.model import ModelDTO
from fedn.network.storage.statestore.stores.dto.round import RoundDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester



@pytest.fixture
def test_rounds():

    return [RoundDTO(round_id=str(uuid.uuid4()), status="test_status6"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status4"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status3"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status1"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status2"),
            RoundDTO(round_id=str(uuid.uuid4()), status="test_status5")]

@pytest.fixture
def test_model_round():
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
    return model, round

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

class TestRoundStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.round_store
    
    @pytest.fixture
    def test_dto(self, db_connection,  test_model_round: RoundDTO):
        model, test_round = test_model_round
        db_connection.model_store.add(model)
        yield test_round
        db_connection.model_store.delete(model.model_id)

    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_rounds: List[RoundDTO]):
        yield from self.helper_prepare_and_clenup(all_db_connections, "round_store", test_rounds)

    def update_function(self, dto: RoundDTO):
        dto.round_config.aggregator = "new_aggregator"
        yield dto
        new_combiner = dto.combiners[0]
        new_combiner.status = "new_status"
        dto.combiners.append(new_combiner)
        yield dto
        dto.round_config = None  # Remove config
        yield dto
        dto.combiners = []  # Remove all combiners
        yield dto

