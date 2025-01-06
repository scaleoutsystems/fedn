from typing import Any, Dict, List, Optional, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import ForeignKey, String, func, select
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from fedn.network.storage.statestore.stores.sql_models import RoundCombinerModel, RoundConfigModel, RoundDataModel, RoundModel
from fedn.network.storage.statestore.stores.store import MongoDBStore, MyAbstractBase, Session, SQLStore, Store

from .shared import EntityNotFound, from_document


class Round:
    def __init__(self, id: str, round_id: str, status: str, round_config: dict, combiners: List[dict], round_data: dict):
        self.id = id
        self.round_id = round_id
        self.status = status
        self.round_config = round_config
        self.combiners = combiners
        self.round_data = round_data


class RoundStore(Store[Round]):
    pass


class MongoDBRoundStore(MongoDBStore[Round]):
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
            raise EntityNotFound(f"Entity with (id | model) {id} not found")

        return from_document(document)

    def update(self, id: str, item: Round) -> bool:
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


def from_row(row: dict) -> Round:
    return {
        "id": row.id,
        "committed_at": row.committed_at,
        "address": row.address,
        "ip": row.ip,
        "name": row.name,
        "parent": row.parent,
        "fqdn": row.fqdn,
        "port": row.port,
        "updated_at": row.updated_at,
    }


class SQLRoundStore(RoundStore, SQLStore[Round]):
    def get(self, id: str) -> Round:
        raise NotImplementedError

    def update(self, id, item):
        raise NotImplementedError

    def add(self, item: Round) -> Tuple[bool, Any]:
        with Session() as session:
            round_id = item["round_id"]
            stmt = select(RoundModel).where(RoundModel.round_id == round_id)
            item = session.scalars(stmt).first()

            if item is not None:
                return False, "Round with round_id already exists"

            round_data = RoundDataModel(
                time_commit=item["round_data"]["time_commit"],
                reduce_time_aggregate_models=item["round_data"]["reduce_time_aggregate_models"],
                reduce_time_fetch_models=item["round_data"]["reduce_time_fetch_models"],
            )

            round_config = RoundConfigModel(
                aggregator=item["round_config"]["aggregator"],
                round_timeout=item["round_config"]["round_timeout"],
                buffer_size=item["round_config"]["buffer_size"],
                delete_models_storage=item["round_config"]["delete_models_storage"],
                clients_required=item["round_config"]["clients_required"],
                validate=item["round_config"]["validate"],
                helper_type=item["round_config"]["helper_type"],
                task=item["round_config"]["task"],
            )

            combiners = []

            for combiner in item["combiners"]:
                combiners.append(
                    RoundCombinerModel(
                        committed_at=combiner["committed_at"],
                        address=combiner["address"],
                        ip=combiner["ip"],
                        name=combiner["name"],
                        parent=combiner["parent"],
                        fqdn=combiner["fqdn"],
                        port=combiner["port"],
                        updated_at=combiner["updated_at"],
                    )
                )

            entity = RoundModel(
                round_id=item["round_id"],
                status=item["status"],
                round_config=round_config,
                combiners=combiners,
                round_data=round_data,
            )

            session.add(entity)
            session.commit()

            return True, from_row(entity)

    def delete(self, id: str) -> bool:
        raise NotImplementedError

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        raise NotImplementedError

    def count(self, **kwargs):
        with Session() as session:
            stmt = select(func.count()).select_from(RoundModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(RoundModel, key) == value)

            count = session.scalar(stmt)

            return count
