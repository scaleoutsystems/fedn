from abc import abstractmethod
from typing import Dict, List

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.round import RoundDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import (
    RoundCombinerDataModel,
    RoundCombinerModel,
    RoundConfigModel,
    RoundDataModel,
    RoundModel,
    from_orm_model,
)
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class RoundStore(Store[RoundDTO]):
    @abstractmethod
    def get_latest_round_id(self) -> int:
        pass


class MongoDBRoundStore(RoundStore, MongoDBStore[RoundDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "round_id")

    def update(self, item: RoundDTO) -> RoundDTO:
        return self.mongo_update(item)

    def get_latest_round_id(self) -> int:
        obj = self.database[self.collection].find_one(sort=[("_id", pymongo.DESCENDING)])
        if obj:
            return int(obj["round_id"])
        else:
            return 0

    def _document_from_dto(self, item: RoundDTO) -> Dict:
        return item.to_db(exclude_unset=False)

    def _dto_from_document(self, document: Dict) -> RoundDTO:
        return RoundDTO().patch_with(from_document(document), throw_on_extra_keys=False)


class SQLRoundStore(RoundStore, SQLStore[RoundDTO, RoundModel]):
    def __init__(self, Session):
        super().__init__(Session, RoundModel, "round_id")

    def update(self, item: RoundDTO) -> RoundDTO:
        return self.sql_update(item)

    def get_latest_round_id(self) -> int:
        rounds = self.list(limit=1, skip=0, sort_key="round_id", sort_order=SortOrder.DESCENDING)
        if any(rounds):
            return int(rounds[0].round_id)
        else:
            return 0

    def _update_orm_model_from_dto(self, entity: RoundModel, item: RoundDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("round_id", None)

        entity.id = item_dict["id"]
        entity.status = item_dict["status"]

        if "round_config" in item_dict and item_dict["round_config"] is not None:
            round_config = item_dict.pop("round_config")
            if entity.round_config is None:
                entity.round_config = RoundConfigModel(**round_config)
            else:
                self._patch_model(entity.round_config, round_config)
        else:
            entity.round_config = None

        if "round_data" in item_dict and item_dict["round_data"] is not None:
            round_data: Dict = item_dict.pop("round_data")
            reduce: Dict = round_data.pop("reduce", {})
            round_data["reduce_time_aggregate_model"] = reduce.pop("time_aggregate_model", None)
            round_data["reduce_time_fetch_model"] = reduce.pop("time_fetch_model", None)
            round_data["reduce_time_load_model"] = reduce.pop("time_load_model", None)
            if entity.round_data is None:
                entity.round_data = RoundDataModel(**round_data)
            else:
                self._patch_model(entity.round_data, round_data)
        else:
            entity.round_data = None

        round_combiners: List[Dict] = item_dict.pop("combiners")
        if round_combiners == self._combiners_to_dict_list(entity.combiners):
            # No changes do nothing
            pass
        else:
            entity.combiners.clear()
            for combiner in round_combiners:
                combiner_round_config: Dict = combiner.pop("config")
                combiner["config_job_id"] = combiner_round_config.pop("_job_id", None)

                combiner_round_config_model = RoundConfigModel(**combiner_round_config)

                combiner_data: Dict = combiner.pop("data")
                aggregation_time = combiner_data.pop("aggregation_time")
                combiner_data_model = RoundCombinerDataModel(**combiner_data)
                combiner_data_model.aggregation_time_nr_aggregated_models = aggregation_time["nr_aggregated_models"]
                combiner_data_model.aggregation_time_time_model_aggregation = aggregation_time["time_model_aggregation"]
                combiner_data_model.aggregation_time_time_model_load = aggregation_time["time_model_load"]

                round_combiner = RoundCombinerModel(**combiner)
                round_combiner.round_config = combiner_round_config_model
                round_combiner.data = combiner_data_model
                entity.combiners.append(round_combiner)

        return entity

    def _dto_from_orm_model(self, item: RoundModel) -> RoundDTO:
        orm_dict = from_orm_model(item, RoundModel)
        if item.round_config is not None:
            orm_dict["round_config"] = from_orm_model(item.round_config, RoundConfigModel)
            del orm_dict["round_config"]["id"]
            del orm_dict["round_config"]["committed_at"]

        if item.combiners is not None:
            orm_dict["combiners"] = self._combiners_to_dict_list(item.combiners)

        if item.round_data is not None:
            orm_dict["round_data"] = from_orm_model(item.round_data, RoundDataModel)
            orm_dict["round_data"]["reduce"] = {
                "time_aggregate_model": orm_dict["round_data"].pop("reduce_time_aggregate_model"),
                "time_fetch_model": orm_dict["round_data"].pop("reduce_time_fetch_model"),
                "time_load_model": orm_dict["round_data"].pop("reduce_time_load_model"),
            }
            del orm_dict["round_data"]["id"]
            del orm_dict["round_data"]["committed_at"]

        orm_dict["round_id"] = orm_dict.pop("id")
        del orm_dict["round_config_id"]
        del orm_dict["round_data_id"]

        return RoundDTO().populate_with(orm_dict)

    def _combiners_to_dict_list(self, combiners: List[RoundCombinerModel]):
        dict_combiners = []
        for c in combiners:
            c_dict = from_orm_model(c, RoundCombinerModel)
            if c.data is not None:
                c_dict["data"] = from_orm_model(c.data, RoundCombinerDataModel)
                c_dict["data"]["aggregation_time"] = {
                    "nr_aggregated_models": c_dict["data"].pop("aggregation_time_nr_aggregated_models"),
                    "time_model_aggregation": c_dict["data"].pop("aggregation_time_time_model_aggregation"),
                    "time_model_load": c_dict["data"].pop("aggregation_time_time_model_load"),
                }
                del c_dict["data"]["id"]
                del c_dict["data"]["committed_at"]

            if c.round_config is not None:
                c_dict["config"] = from_orm_model(c.round_config, RoundConfigModel)
                c_dict["config"]["_job_id"] = c_dict.pop("config_job_id")
                del c_dict["config"]["id"]
                del c_dict["config"]["committed_at"]

            del c_dict["id"]
            del c_dict["committed_at"]
            del c_dict["round_config_id"]
            del c_dict["data_id"]
            del c_dict["parent_round_id"]

            dict_combiners.append(c_dict)
        return dict_combiners

    def _patch_model(self, model, item_dict: Dict):
        for key, value in item_dict.items():
            setattr(model, key, value)
        return model
