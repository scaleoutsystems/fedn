from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import Integer, func, or_, select

from fedn.network.storage.statestore.stores.sql.shared import RoundCombinerModel, RoundConfigModel, RoundDataModel, RoundModel
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store

from .shared import from_document


class Round:
    def __init__(self, id: str, round_id: str, status: str, round_config: dict, combiners: List[dict], round_data: dict):
        self.id = id
        self.round_id = round_id
        self.status = status
        self.round_config = round_config
        self.combiners = combiners
        self.round_data = round_data


class RoundStore(Store[Round]):
    @abstractmethod
    def get_latest_round_id(self) -> int:
        pass


class MongoDBRoundStore(RoundStore, MongoDBStore[Round]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index([("round_id", pymongo.DESCENDING)])

    def get(self, id: str) -> Round:
        """Get an entity by id
        param id: The id of the entity
            type: str
        return: The entity
        """
        kwargs = {}
        if ObjectId.is_valid(id):
            id_obj = ObjectId(id)
            kwargs["_id"] = id_obj
        else:
            kwargs["round_id"] = id

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            return None

        return from_document(document)

    def update(self, id: str, item: Round) -> Tuple[bool, Any]:
        return super().update(id, item)

    def add(self, item: Round) -> Tuple[bool, Any]:
        round_id = item["round_id"]
        existing = self.database[self.collection].find_one({"round_id": round_id})

        if existing is not None:
            return False, "Round with round_id already exists"

        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for RoundStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Round]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
            description: The maximum number of entities to return
        param skip: The number of entities to skip
            type: int
            description: The number of entities to skip
        param sort_key: The key to sort by
            type: str
            description: The key to sort by
        param sort_order: The order to sort by
            type: pymongo.DESCENDING
            description: The order to sort by
        return: The entities
        """
        return super().list(limit, skip, sort_key or "round_id", sort_order, **kwargs)

    def get_latest_round_id(self) -> int:
        obj = self.database[self.collection].find_one(sort=[("_id", pymongo.DESCENDING)])
        if obj:
            return int(obj["round_id"])
        else:
            return 0


def from_row(row: RoundModel) -> Round:
    round_data = None
    if row.round_data is not None:
        round_data = {
            "time_commit": row.round_data.time_commit,
            "reduce": {
                "time_aggregate_model": row.round_data.reduce_time_aggregate_model,
                "time_fetch_model": row.round_data.reduce_time_fetch_model,
                "time_load_model": row.round_data.reduce_time_load_model,
            },
        }

    round_config = None

    if row.round_config is not None:
        round_config = {
            "aggregator": row.round_config.aggregator,
            "round_timeout": row.round_config.round_timeout,
            "buffer_size": row.round_config.buffer_size,
            "delete_models_storage": row.round_config.delete_models_storage,
            "clients_required": row.round_config.clients_required,
            "validate": row.round_config.validate,
            "helper_type": row.round_config.helper_type,
            "task": row.round_config.task,
            "model_id": row.round_config.model_id,
            "session_id": row.round_config.session_id,
            "round_id": row.round_config.round_id,
            "rounds": row.round_config.rounds,
        }

    combiners = [
        {
            "model_id": combiner.model_id,
            "name": combiner.name,
            "round_id": combiner.round_id,
            "status": combiner.status,
            "time_exec_training": combiner.time_exec_training,
            "config": {
                "_job_id": combiner.config__job_id,
                "aggregator": combiner.config_aggregator,
                "buffer_size": combiner.config_buffer_size,
                "clients_required": combiner.config_clients_required,
                "delete_models_storage": combiner.config_delete_models_storage,
                "helper_type": combiner.config_helper_type,
                "model_id": combiner.config_model_id,
                "round_id": combiner.config_round_id,
                "round_timeout": combiner.config_round_timeout,
                "rounds": combiner.config_rounds,
                "session_id": combiner.config_session_id,
                "task": combiner.config_task,
                "validate": combiner.config_validate,
            },
            "data": {
                "aggregation_time": {
                    "nr_aggregated_models": combiner.data_aggregation_time_nr_aggregated_models,
                    "time_model_aggregation": combiner.data_aggregation_time_time_model_aggregation,
                    "time_model_load": combiner.data_aggregation_time_time_model_load,
                },
                "nr_expected_updates": combiner.data_nr_expected_updates,
                "nr_required_updates": combiner.data_nr_required_updates,
                "time_combination": combiner.data_time_combination,
                "timeout": combiner.data_timeout,
            },
        }
        for combiner in row.combiners
    ]

    return {
        "id": row.id,
        "committed_at": row.committed_at,
        "round_id": row.round_id,
        "status": row.status,
        "round_config": round_config,
        "round_data": round_data,
        "combiners": combiners,
    }


class SQLRoundStore(RoundStore, SQLStore[Round]):
    def __init__(self, Session):
        super().__init__(Session)

    def get(self, id: str) -> Round:
        with self.Session() as session:
            stmt = select(RoundModel).where(or_(RoundModel.id == id, RoundModel.round_id == id))
            item = session.scalars(stmt).first()

            if item is None:
                return None

            return from_row(item)

    def update(self, id, item: Round) -> Tuple[bool, Any]:
        with self.Session() as session:
            stmt = select(RoundModel).where(or_(RoundModel.id == id, RoundModel.round_id == id))
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                return False, f"Entity with (id | round_id) {id} not found"

            if "round_data" in item and item["round_data"] is not None:
                round_data = item["round_data"]
                reduce = round_data["reduce"] if "reduce" in round_data else {}

                if existing_item.round_data is None:
                    round_data = RoundDataModel(
                        time_commit=round_data["time_commit"] if "time_commit" in round_data else None,
                        reduce_time_aggregate_model=reduce["time_aggregate_model"] if "time_aggregate_model" in reduce else None,
                        reduce_time_fetch_model=reduce["time_fetch_model"] if "time_fetch_model" in reduce else None,
                        reduce_time_load_model=reduce["time_load_model"] if "time_load_model" in reduce else None,
                    )
                    session.add(round_data)
                    existing_item.round_data = round_data
                else:
                    existing_item.round_data.time_commit = round_data["time_commit"] if "time_commit" in round_data else None
                    existing_item.round_data.reduce_time_aggregate_model = reduce["time_aggregate_model"] if "time_aggregate_model" in reduce else None
                    existing_item.round_data.reduce_time_fetch_model = reduce["time_fetch_model"] if "time_fetch_model" in reduce else None
                    existing_item.round_data.reduce_time_load_model = reduce["time_load_model"] if "time_load_model" in reduce else None

            if "round_config" in item and item["round_config"] is not None:
                if existing_item.round_config is None:
                    existing_item.round_config = RoundConfigModel(
                        aggregator=item["round_config"]["aggregator"],
                        round_timeout=item["round_config"]["round_timeout"],
                        buffer_size=item["round_config"]["buffer_size"],
                        delete_models_storage=item["round_config"]["delete_models_storage"],
                        clients_required=item["round_config"]["clients_required"],
                        validate=item["round_config"]["validate"],
                        helper_type=item["round_config"]["helper_type"],
                        task=item["round_config"]["task"],
                        model_id=item["round_config"]["model_id"],
                        session_id=item["round_config"]["session_id"],
                        round_id=item["round_config"]["round_id"],
                        rounds=item["round_config"]["rounds"],
                    )
                else:
                    existing_item.round_config.aggregator = item["round_config"]["aggregator"]
                    existing_item.round_config.round_timeout = item["round_config"]["round_timeout"]
                    existing_item.round_config.buffer_size = item["round_config"]["buffer_size"]
                    existing_item.round_config.delete_models_storage = item["round_config"]["delete_models_storage"]
                    existing_item.round_config.clients_required = item["round_config"]["clients_required"]
                    existing_item.round_config.validate = item["round_config"]["validate"]
                    existing_item.round_config.helper_type = item["round_config"]["helper_type"]
                    existing_item.round_config.task = item["round_config"]["task"]
                    existing_item.round_config.round_id = item["round_config"]["round_id"]
                    existing_item.round_config.rounds = item["round_config"]["rounds"]

                    if "model_id" in item["round_config"]:
                        existing_item.round_config.model_id = item["round_config"]["model_id"]

                    if "session_id" in item["round_config"]:
                        existing_item.round_config.session_id = item["round_config"]["session_id"]

            if "combiners" in item and item["combiners"] is not None:
                if existing_item.combiners is not None:
                    existing_item.combiners.clear()

                for combiner in item["combiners"]:
                    config = combiner["config"] if "config" in combiner else {}
                    data = combiner["data"] if "data" in combiner else {}
                    aggregation_time = data["aggregation_time"] if "aggregation_time" in data else {}

                    child = RoundCombinerModel(
                        model_id=combiner["model_id"],
                        name=combiner["name"],
                        round_id=combiner["round_id"],
                        status=combiner["status"],
                        time_exec_training=combiner["time_exec_training"],
                        config__job_id=config["_job_id"],
                        config_aggregator=config["aggregator"],
                        config_buffer_size=config["buffer_size"],
                        config_clients_required=config["clients_required"],
                        config_delete_models_storage=config["delete_models_storage"]
                        if isinstance(config["delete_models_storage"], bool)
                        else config["delete_models_storage"] == "True",
                        config_helper_type=config["helper_type"],
                        config_model_id=config["model_id"],
                        config_round_id=config["round_id"],
                        config_round_timeout=config["round_timeout"],
                        config_rounds=config["rounds"],
                        config_session_id=config["session_id"],
                        config_task=config["task"],
                        config_validate=config["validate"] if isinstance(config["validate"], bool) else config["validate"] == "True",
                        data_aggregation_time_nr_aggregated_models=aggregation_time["nr_aggregated_models"]
                        if "nr_aggregated_models" in aggregation_time
                        else None,
                        data_aggregation_time_time_model_aggregation=aggregation_time["time_model_aggregation"]
                        if "time_model_aggregation" in aggregation_time
                        else None,
                        data_aggregation_time_time_model_load=aggregation_time["time_model_load"] if "time_model_load" in aggregation_time else None,
                        data_nr_expected_updates=data["nr_expected_updates"] if "nr_expected_updates" in data else None,
                        data_nr_required_updates=data["nr_required_updates"] if "nr_required_updates" in data else None,
                        data_time_combination=data["time_combination"] if "time_combination" in data else None,
                        data_timeout=data["timeout"] if "timeout" in data else None,
                    )

                    existing_item.combiners.append(child)

            existing_item.status = item["status"]
            session.commit()

            return True, from_row(existing_item)

    def add(self, item: Round) -> Tuple[bool, Any]:
        with self.Session() as session:
            round_id = item["round_id"]
            stmt = select(RoundModel).where(RoundModel.round_id == round_id)
            existing_item = session.scalars(stmt).first()

            if existing_item is not None:
                return False, "Round with round_id already exists"

            entity = RoundModel(round_id=item["round_id"], status=item["status"])

            round_data: RoundDataModel = None
            if "round_data" in item:
                round_data = RoundDataModel(
                    time_commit=item["round_data"]["time_commit"],
                    reduce_time_aggregate_model=item["round_data"]["reduce"]["time_aggregate_model"],
                    reduce_time_fetch_model=item["round_data"]["reduce"]["time_fetch_model"],
                    reduce_time_load_model=item["round_data"]["reduce"]["time_load_model"],
                )
                entity.round_data = round_data

            round_config: RoundConfigModel = None

            if "round_config" in item:
                round_config = RoundConfigModel(
                    aggregator=item["round_config"]["aggregator"],
                    round_timeout=item["round_config"]["round_timeout"],
                    buffer_size=item["round_config"]["buffer_size"],
                    delete_models_storage=item["round_config"]["delete_models_storage"],
                    clients_required=item["round_config"]["clients_required"],
                    validate=item["round_config"]["validate"],
                    helper_type=item["round_config"]["helper_type"],
                    task=item["round_config"]["task"],
                    model_id=item["round_config"]["model_id"],
                    session_id=item["round_config"]["session_id"],
                    round_id=item["round_config"]["round_id"],
                    rounds=item["round_config"]["rounds"],
                )
                entity.round_config = round_config

            combiners: List[RoundCombinerModel] = []

            if "combiners" in item:
                combiners = []

                for combiner in item["combiners"]:
                    config = combiner["config"] if "config" in combiner else {}
                    data = combiner["data"] if "data" in combiner else {}
                    aggregation_time = data["aggregation_time"] if "aggregation_time" in data else {}

                    combiners.append(
                        RoundCombinerModel(
                            model_id=combiner["model_id"],
                            name=combiner["name"],
                            round_id=combiner["round_id"],
                            status=combiner["status"],
                            time_exec_training=combiner["time_exec_training"],
                            config__job_id=config["_job_id"],
                            config_aggregator=config["aggregator"],
                            config_buffer_size=config["buffer_size"],
                            config_clients_required=config["clients_required"],
                            config_delete_models_storage=config["delete_models_storage"]
                            if isinstance(config["delete_models_storage"], bool)
                            else config["delete_models_storage"] == "True",
                            config_helper_type=config["helper_type"],
                            config_model_id=config["model_id"],
                            config_round_id=config["round_id"],
                            config_round_timeout=config["round_timeout"],
                            config_rounds=config["rounds"],
                            config_session_id=config["session_id"],
                            config_task=config["task"],
                            config_validate=config["validate"] if isinstance(config["validate"], bool) else config["validate"] == "True",
                            data_aggregation_time_nr_aggregated_models=aggregation_time["nr_aggregated_models"]
                            if "nr_aggregated_models" in aggregation_time
                            else None,
                            data_aggregation_time_time_model_aggregation=aggregation_time["time_model_aggregation"]
                            if "time_model_aggregation" in aggregation_time
                            else None,
                            data_aggregation_time_time_model_load=aggregation_time["time_model_load"] if "time_model_load" in aggregation_time else None,
                            data_nr_expected_updates=data["nr_expected_updates"] if "nr_expected_updates" in data else None,
                            data_nr_required_updates=data["nr_required_updates"] if "nr_required_updates" in data else None,
                            data_time_combination=data["time_combination"] if "time_combination" in data else None,
                            data_timeout=data["timeout"] if "timeout" in data else None,
                        )
                    )

                entity.combiners = combiners

            session.add(entity)
            session.commit()

            return True, from_row(entity)

    def delete(self, id: str) -> bool:
        raise NotImplementedError

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        with self.Session() as session:
            stmt = select(RoundModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(RoundModel, key) == value)

            if sort_key:
                _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
                _sort_key: str = sort_key or "committed_at"

                if _sort_key == "_id":
                    _sort_key = "id"

                if _sort_key == "round_id":
                    sort_obj = RoundModel.round_id.cast(Integer) if _sort_order == "ASC" else RoundModel.round_id.cast(Integer).desc()
                elif _sort_key in RoundModel.__table__.columns:
                    sort_obj = RoundModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else RoundModel.__table__.columns.get(_sort_key).desc()

                stmt = stmt.order_by(sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            if skip:
                stmt = stmt.offset(skip)

            items = session.execute(stmt)

            result = []

            for item in items:
                (r,) = item

                result.append(from_row(r))

            return {"count": len(result), "result": result}

    def count(self, **kwargs):
        with self.Session() as session:
            stmt = select(func.count()).select_from(RoundModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(RoundModel, key) == value)

            count = session.scalar(stmt)

            return count

    def get_latest_round_id(self) -> int:
        response = self.list(limit=1, skip=0, sort_key="round_id", sort_order=pymongo.DESCENDING)
        if response and "result" in response and len(response["result"]) > 0:
            round_id: str = response["result"][0]["round_id"]
            return int(round_id)
        else:
            return 0
