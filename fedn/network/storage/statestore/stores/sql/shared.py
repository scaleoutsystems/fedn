import uuid
from datetime import datetime
from typing import List, Optional, Type

from sqlalchemy import ForeignKey, MetaData, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

constraint_naming_conventions = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def from_orm_model(model, SQLModel: Type[DeclarativeBase]):
    result = {}
    for k in SQLModel.__table__.columns:
        result[k.name] = getattr(model, k.name)
    return result


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=constraint_naming_conventions)


class MyAbstractBase(Base):
    __abstract__ = True

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    committed_at: Mapped[datetime] = mapped_column(default=datetime.now)


class SessionConfigModel(MyAbstractBase):
    __tablename__ = "session_configs"

    aggregator: Mapped[str] = mapped_column(String(255))
    round_timeout: Mapped[int]
    buffer_size: Mapped[int]
    delete_models_storage: Mapped[bool]
    clients_required: Mapped[int]
    validate: Mapped[bool]
    helper_type: Mapped[str] = mapped_column(String(255))
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    session: Mapped["SessionModel"] = relationship(back_populates="session_config")
    rounds: Mapped[int]
    server_functions: Mapped[Optional[str]] = mapped_column(String(255))


class SessionModel(MyAbstractBase):
    __tablename__ = "sessions"

    name: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
    session_config_id: Mapped[str] = mapped_column(ForeignKey("session_configs.id"))
    session_config: Mapped["SessionConfigModel"] = relationship(back_populates="session")
    models: Mapped[List["ModelModel"]] = relationship(back_populates="session")


class ModelModel(MyAbstractBase):
    __tablename__ = "models"

    active: Mapped[bool] = mapped_column(default=False)
    parent_model: Mapped[Optional[str]] = mapped_column(String(255))
    name: Mapped[Optional[str]] = mapped_column(String(255))
    session_configs: Mapped[List["SessionConfigModel"]] = relationship()
    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    session: Mapped[Optional["SessionModel"]] = relationship(back_populates="models")


class RoundConfigModel(MyAbstractBase):
    __tablename__ = "round_configs"

    aggregator: Mapped[str] = mapped_column(String(255))
    round_timeout: Mapped[int]
    buffer_size: Mapped[int]
    delete_models_storage: Mapped[bool]
    clients_required: Mapped[int]
    validate: Mapped[bool]
    helper_type: Mapped[str] = mapped_column(String(255))
    model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    round: Mapped["RoundModel"] = relationship(back_populates="round_config")
    task: Mapped[str] = mapped_column(String(255))
    round_id: Mapped[str]
    rounds: Mapped[int]


class RoundDataModel(MyAbstractBase):
    __tablename__ = "round_data"

    time_commit: Mapped[Optional[float]]
    reduce_time_aggregate_model: Mapped[Optional[float]]
    reduce_time_fetch_model: Mapped[Optional[float]]
    reduce_time_load_model: Mapped[Optional[float]]
    round: Mapped["RoundModel"] = relationship(back_populates="round_data")


class RoundCombinerModel(MyAbstractBase):
    __tablename__ = "round_combiners"

    model_id: Mapped[str]
    name: Mapped[str] = mapped_column(String(255))
    round_id: Mapped[str]
    status: Mapped[str] = mapped_column(String(255))
    time_exec_training: Mapped[float]

    config__job_id: Mapped[str] = mapped_column(String(255))
    config_aggregator: Mapped[str] = mapped_column(String(255))
    config_buffer_size: Mapped[int]
    config_clients_required: Mapped[int]
    config_delete_models_storage: Mapped[bool]
    config_helper_type: Mapped[str] = mapped_column(String(255))
    config_model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    config_round_id: Mapped[str]
    config_round_timeout: Mapped[int]
    config_rounds: Mapped[int]
    config_session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    config_task: Mapped[str] = mapped_column(String(255))
    config_validate: Mapped[bool]

    data_aggregation_time_nr_aggregated_models: Mapped[Optional[int]]
    data_aggregation_time_time_model_aggregation: Mapped[Optional[float]]
    data_aggregation_time_time_model_load: Mapped[Optional[float]]
    data_nr_expected_updates: Mapped[Optional[int]]
    data_nr_required_updates: Mapped[Optional[int]]
    data_time_combination: Mapped[Optional[float]]
    data_timeout: Mapped[Optional[float]]

    parent_round_id: Mapped[str] = mapped_column(ForeignKey("rounds.id"))
    round: Mapped["RoundModel"] = relationship(back_populates="combiners")


class RoundModel(MyAbstractBase):
    __tablename__ = "rounds"

    round_id: Mapped[str] = mapped_column(unique=True)  # TODO: Add unique constraint. Does this work?
    status: Mapped[str] = mapped_column()

    round_config_id: Mapped[Optional[str]] = mapped_column(ForeignKey("round_configs.id"))
    round_config: Mapped[Optional["RoundConfigModel"]] = relationship(back_populates="round")
    round_data_id: Mapped[Optional[str]] = mapped_column(ForeignKey("round_data.id"))
    round_data: Mapped[Optional["RoundDataModel"]] = relationship(back_populates="round")
    combiners: Mapped[List["RoundCombinerModel"]] = relationship(back_populates="round", cascade="all, delete-orphan")


class ClientModel(MyAbstractBase):
    __tablename__ = "clients"

    combiner: Mapped[str] = mapped_column(String(255))
    combiner_preferred: Mapped[str] = mapped_column(String(255))
    ip: Mapped[Optional[str]] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    package: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
    last_seen: Mapped[datetime] = mapped_column(default=datetime.now())


class PackageModel(MyAbstractBase):
    __tablename__ = "packages"

    active: Mapped[bool] = mapped_column(default=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    file_name: Mapped[str] = mapped_column(String(255))
    helper: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    storage_file_name: Mapped[str] = mapped_column(String(255))


class CombinerModel(MyAbstractBase):
    __tablename__ = "combiners"

    address: Mapped[str] = mapped_column(String(255))
    fqdn: Mapped[Optional[str]] = mapped_column(String(255))
    ip: Mapped[Optional[str]] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    parent: Mapped[Optional[str]] = mapped_column(String(255))
    port: Mapped[int]
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now())


class PredictionModel(MyAbstractBase):
    __tablename__ = "predictions"

    correlation_id: Mapped[str]
    data: Mapped[Optional[str]]
    model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    receiver_name: Mapped[Optional[str]] = mapped_column(String(255))
    receiver_role: Mapped[Optional[str]] = mapped_column(String(255))
    sender_name: Mapped[Optional[str]] = mapped_column(String(255))
    sender_role: Mapped[Optional[str]] = mapped_column(String(255))
    meta: Mapped[Optional[str]] = mapped_column(String(255))
    timestamp: Mapped[str] = mapped_column(String(255))


class StatusModel(MyAbstractBase):
    __tablename__ = "statuses"

    log_level: Mapped[str] = mapped_column(String(255))
    sender_name: Mapped[Optional[str]] = mapped_column(String(255))
    sender_role: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[datetime]
    type: Mapped[str] = mapped_column(String(255))
    data: Mapped[Optional[str]]
    correlation_id: Mapped[Optional[str]]
    extra: Mapped[Optional[str]]
    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
