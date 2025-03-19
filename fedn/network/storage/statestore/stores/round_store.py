import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database
from sqlalchemy import select

from fedn.network.storage.statestore.stores.dto.round import RoundDTO
from fedn.network.storage.statestore.stores.new_store import MongoDBStore, SQLStore, Store, from_document
from fedn.network.storage.statestore.stores.sql.shared import (
    RoundCombinerDataModel,
    RoundCombinerModel,
    RoundConfigModel,
    RoundDataModel,
    RoundModel,
    from_orm_model,
)


class RoundStore(Store[RoundDTO]):
    @abstractmethod
    def get_latest_round_id(self) -> int:
        pass


class MongoDBRoundStore(RoundStore, MongoDBStore):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "round_id")

    def get(self, id: str) -> RoundDTO:
        document = self.mongo_get(id)
        if document is None:
            return None
        return self._dto_from_document(document)

    def update(self, item: RoundDTO) -> Tuple[bool, Any]:
        document = item.to_db(exclude_unset=True)

        success, obj = self.mongo_update(document)

        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def add(self, item: RoundDTO) -> Tuple[bool, Any]:
        document = item.to_db(exclude_unset=False)
        success, obj = self.mongo_add(document)
        if success:
            return success, self._dto_from_document(obj)
        return success, obj

    def delete(self, id: str) -> bool:
        return self.mongo_delete(id)

    def select(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=pymongo.DESCENDING, **filter_kwargs) -> List[RoundDTO]:
        entities = self.mongo_select(limit, skip, sort_key, sort_order, **filter_kwargs)
        return [self._dto_from_document(entity) for entity in entities]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[RoundDTO]]:
        raise NotImplementedError("Deprecated method, use select instead")

    def get_latest_round_id(self) -> int:
        obj = self.database[self.collection].find_one(sort=[("committed_at", pymongo.DESCENDING)])
        if obj:
            return int(obj["round_id"])
        else:
            return 0

    def count(self, **kwargs) -> int:
        return self.mongo_count(**kwargs)

    def _dto_from_document(self, document: Dict) -> RoundDTO:
        return RoundDTO().populate_with(from_document(document))


class SQLRoundStore(RoundStore, SQLStore[RoundDTO]):
    def __init__(self, Session):
        super().__init__(Session, RoundModel)

    def get(self, id: str) -> RoundDTO:
        with self.Session() as session:
            stmt = select(RoundModel).where(RoundModel.id == id)
            item = session.scalars(stmt).first()
            if item is None:
                return None
            return self._dto_from_orm_model(item)

    def add(self, item: RoundDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            item_dict = item.to_db()

            round_entity = RoundModel()
            if "round_id" in item_dict:
                round_entity.id = item_dict["round_id"]
            else:
                round_entity.id = str(uuid.uuid4())
            round_entity.status = item_dict["status"]

            if "round_config" in item_dict and item_dict["round_config"] is not None:
                round_config_entity = RoundConfigModel(**item_dict["round_config"])
                del item_dict["round_config"]
                round_entity.round_config = round_config_entity

            if "round_data" in item_dict and item_dict["round_data"] is not None:
                round_data = item_dict.pop("round_data")
                if "reduce" in round_data:
                    reduce = round_data.pop("reduce")
                    round_data_entity = RoundDataModel(**round_data)
                    round_data_entity.reduce_time_aggregate_model = reduce["time_aggregate_model"]
                    round_data_entity.reduce_time_fetch_model = reduce["time_fetch_model"]
                    round_data_entity.reduce_time_load_model = reduce["time_load_model"]
                else:
                    round_data_entity = RoundDataModel(**round_data)
                round_entity.round_data = round_data_entity

            if "combiners" in item_dict and item_dict["combiners"] is not None:
                round_combiners = item_dict.pop("combiners")
                for combiner in round_combiners:
                    combiner_round_config = combiner.pop("config")
                    combiner["config_job_id"] = combiner_round_config.pop("_job_id", None)

                    combiner_round_config_model = RoundConfigModel(**combiner_round_config)

                    combiner_data = combiner.pop("data")
                    aggregation_time = combiner_data.pop("aggregation_time")
                    combiner_data_model = RoundCombinerDataModel(**combiner_data)
                    combiner_data_model.aggregation_time_nr_aggregated_models = aggregation_time["nr_aggregated_models"]
                    combiner_data_model.aggregation_time_time_model_aggregation = aggregation_time["time_model_aggregation"]
                    combiner_data_model.aggregation_time_time_model_load = aggregation_time["time_model_load"]

                    round_combiner = RoundCombinerModel(**combiner)
                    round_combiner.round_config = combiner_round_config_model
                    round_combiner.data = combiner_data_model
                    round_entity.combiners.append(round_combiner)
            session.add(round_entity)
            session.commit()
            return True, self._dto_from_orm_model(round_entity)

    def update(self, item: RoundDTO) -> Tuple[bool, Any]:
        with self.Session() as session:
            stmt = select(RoundModel).where(RoundModel.id == item.round_id)
            existing_item = session.scalars(stmt).first()

            item_dict = item.to_db()

            if existing_item is None:
                return False, f"Entity with (id | round_id) {item.round_id} not found"

            if item_dict["round_data"] is not None:
                round_data = item_dict["round_data"]
                reduce = round_data["reduce"] if "reduce" in round_data else {}

                if existing_item.round_data is None:
                    round_data_model = RoundDataModel(
                        time_commit=round_data["time_commit"] if "time_commit" in round_data else None,
                        reduce_time_aggregate_model=reduce["time_aggregate_model"] if "time_aggregate_model" in reduce else None,
                        reduce_time_fetch_model=reduce["time_fetch_model"] if "time_fetch_model" in reduce else None,
                        reduce_time_load_model=reduce["time_load_model"] if "time_load_model" in reduce else None,
                    )
                    session.add(round_data_model)
                    existing_item.round_data = round_data_model
                else:
                    existing_item.round_data.time_commit = round_data["time_commit"] if "time_commit" in round_data else None
                    existing_item.round_data.reduce_time_aggregate_model = reduce["time_aggregate_model"] if "time_aggregate_model" in reduce else None
                    existing_item.round_data.reduce_time_fetch_model = reduce["time_fetch_model"] if "time_fetch_model" in reduce else None
                    existing_item.round_data.reduce_time_load_model = reduce["time_load_model"] if "time_load_model" in reduce else None
            elif existing_item.round_data is not None:
                session.delete(existing_item.round_data)
                existing_item.round_data = None

            if item_dict["round_config"] is not None:
                if existing_item.round_config is None:
                    existing_item.round_config = RoundConfigModel(**item_dict["round_config"])
                else:
                    for key, value in item_dict["round_config"].items():
                        setattr(existing_item.round_config, key, value)
            elif existing_item.round_config is not None:
                session.delete(existing_item.round_config)
                existing_item.round_config = None

            if item_dict["combiners"] is not None:
                existing_item.combiners.clear()

                for combiner in item_dict["combiners"]:
                    config = combiner.pop("config", {})
                    data = combiner.pop("data", {})
                    aggregation_time = data.pop("aggregation_time", {})
                    data["aggregation_time_nr_aggregated_models"] = aggregation_time.pop("nr_aggregated_models", None)
                    data["aggregation_time_time_model_aggregation"] = aggregation_time.pop("time_model_aggregation", None)
                    data["aggregation_time_time_model_load"] = aggregation_time.pop("time_model_load", None)

                    combiner["config_job_id"] = config.pop("_job_id", None)

                    child = RoundCombinerModel(**combiner)
                    config_model = RoundConfigModel(**config)
                    child.round_config = config_model
                    data_model = RoundCombinerDataModel(**data)
                    child.data = data_model

                    existing_item.combiners.append(child)

            existing_item.status = item_dict["status"]
            session.commit()

            return True, self._dto_from_orm_model(existing_item)

    def delete(self, id: str) -> bool:
        return self.sql_delete(id)

    def select(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            enties = self.sql_select(session, limit, skip, sort_key, sort_order, **kwargs)
            return [self._dto_from_orm_model(entity) for entity in enties]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        raise NotImplementedError("Deprecated method, use select instead")

    def count(self, **kwargs):
        return self.sql_count(**kwargs)

    def get_latest_round_id(self) -> int:
        rounds = self.select(limit=1, skip=0, sort_key="round_id", sort_order=pymongo.DESCENDING)
        if any(rounds):
            return int(rounds[0].round_id)
        else:
            return 0

    def _orm_dict_from_dto_dict(self, item_dict: Dict) -> Dict:
        item_dict["id"] = item_dict.pop("round_id")
        return item_dict

    def _dto_from_orm_model(self, item: RoundModel) -> RoundDTO:
        orm_dict = from_orm_model(item, RoundModel)
        if item.round_config is not None:
            orm_dict["round_config"] = from_orm_model(item.round_config, RoundConfigModel)
            del orm_dict["round_config"]["id"]
            del orm_dict["round_config"]["committed_at"]

        if item.combiners is not None:
            orm_dict["combiners"] = []
            for c in item.combiners:
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

                orm_dict["combiners"].append(c_dict)

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
