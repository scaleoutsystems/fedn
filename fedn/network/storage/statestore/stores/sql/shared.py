import uuid
from datetime import datetime
from typing import Dict, List, Optional, Type

from sqlalchemy import JSON, ForeignKey, MetaData, String
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
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now, onupdate=datetime.now)


class SessionConfigModel(MyAbstractBase):
    __tablename__ = "session_configs"

    aggregator: Mapped[str] = mapped_column(String(255))
    aggregator_kwargs: Mapped[Optional[str]]
    round_timeout: Mapped[int]
    buffer_size: Mapped[int]
    delete_models_storage: Mapped[bool]
    clients_required: Mapped[int]
    requested_clients: Mapped[Optional[int]]
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
    session_config: Mapped["SessionConfigModel"] = relationship(back_populates="session", cascade="all, delete-orphan", single_parent=True)

    models: Mapped[List["ModelModel"]] = relationship(back_populates="session", foreign_keys="ModelModel.session_id")

    seed_model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    seed_model: Mapped[Optional["ModelModel"]] = relationship(foreign_keys="[SessionModel.seed_model_id]")


class ModelModel(MyAbstractBase):
    __tablename__ = "models"

    active: Mapped[bool] = mapped_column(default=False)
    parent_model: Mapped[Optional[str]] = mapped_column(String(255))
    name: Mapped[Optional[str]] = mapped_column(String(255))

    session_configs: Mapped[List["SessionConfigModel"]] = relationship()

    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    session: Mapped[Optional["SessionModel"]] = relationship(back_populates="models", foreign_keys="[ModelModel.session_id]")


class RoundConfigModel(MyAbstractBase):
    __tablename__ = "round_configs"

    aggregator: Mapped[str] = mapped_column(String(255))
    aggregator_kwargs: Mapped[Optional[str]]
    round_timeout: Mapped[int]
    buffer_size: Mapped[int]
    delete_models_storage: Mapped[bool]
    clients_required: Mapped[int]
    requested_clients: Mapped[Optional[int]]
    validate: Mapped[bool]
    helper_type: Mapped[str] = mapped_column(String(255))
    server_functions: Mapped[Optional[str]]
    model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    task: Mapped[str] = mapped_column(String(255))
    round_id: Mapped[str]
    rounds: Mapped[int]
    client_settings: Mapped[Optional[Dict]] = mapped_column(JSON)
    is_sl_inference: Mapped[bool]


class RoundCombinerDataModel(MyAbstractBase):
    __tablename__ = "round_combiner_data"

    aggregation_time_nr_aggregated_models: Mapped[Optional[int]]
    aggregation_time_time_model_aggregation: Mapped[Optional[float]]
    aggregation_time_time_model_load: Mapped[Optional[float]]
    nr_expected_updates: Mapped[Optional[int]]
    nr_required_updates: Mapped[Optional[int]]
    time_combination: Mapped[Optional[float]]
    timeout: Mapped[Optional[float]]
    round_combiner: Mapped["RoundCombinerModel"] = relationship(back_populates="data")


class RoundCombinerModel(MyAbstractBase):
    __tablename__ = "round_combiners"

    model_id: Mapped[str]
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
    time_exec_training: Mapped[float]

    round_config_id: Mapped[Optional[str]] = mapped_column(ForeignKey("round_configs.id"))
    round_config: Mapped[Optional["RoundConfigModel"]] = relationship(cascade="all, delete-orphan", single_parent=True)

    data_id: Mapped[Optional[str]] = mapped_column(ForeignKey("round_combiner_data.id"))
    data: Mapped[Optional["RoundCombinerDataModel"]] = relationship(back_populates="round_combiner", cascade="all, delete")

    parent_round_id: Mapped[str] = mapped_column(ForeignKey("rounds.id"))
    round: Mapped["RoundModel"] = relationship(back_populates="combiners")

    config_job_id: Mapped[Optional[str]]
    round_id: Mapped[Optional[str]]


class RoundDataModel(MyAbstractBase):
    __tablename__ = "round_data"

    time_commit: Mapped[Optional[float]]
    reduce_time_aggregate_model: Mapped[Optional[float]]
    reduce_time_fetch_model: Mapped[Optional[float]]
    reduce_time_load_model: Mapped[Optional[float]]
    round: Mapped["RoundModel"] = relationship(back_populates="round_data")


class RoundModel(MyAbstractBase):
    __tablename__ = "rounds"
    status: Mapped[str] = mapped_column()

    round_config_id: Mapped[Optional[str]] = mapped_column(ForeignKey("round_configs.id", ondelete="cascade"))
    round_config: Mapped[Optional["RoundConfigModel"]] = relationship(cascade="all, delete-orphan", single_parent=True)
    round_data_id: Mapped[Optional[str]] = mapped_column(ForeignKey("round_data.id", ondelete="cascade"))
    round_data: Mapped[Optional["RoundDataModel"]] = relationship(back_populates="round", cascade="all, delete")
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


class ValidationModel(MyAbstractBase):
    __tablename__ = "validations"

    correlation_id: Mapped[str]
    data: Mapped[Optional[str]]
    model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    meta: Mapped[Optional[str]] = mapped_column(String(255))
    receiver_name: Mapped[Optional[str]] = mapped_column(String(255))
    receiver_role: Mapped[Optional[str]] = mapped_column(String(255))
    sender_name: Mapped[Optional[str]] = mapped_column(String(255))
    sender_role: Mapped[Optional[str]] = mapped_column(String(255))
    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    timestamp: Mapped[datetime]


class MetricModel(MyAbstractBase):
    __tablename__ = "metrics"

    key: Mapped[str] = mapped_column(String(255))
    value: Mapped[float]

    # Client timestamp
    timestamp: Mapped[Optional[datetime]]

    sender_name: Mapped[str]
    sender_role: Mapped[str]

    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    step: Mapped[Optional[int]]

    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    round_id: Mapped[Optional[str]] = mapped_column(ForeignKey("rounds.id"))


class AttributeModel(MyAbstractBase):
    __tablename__ = "attributes"

    key: Mapped[str] = mapped_column(String(255))
    value: Mapped[str]

    # Client timestamp
    timestamp: Mapped[Optional[datetime]]

    sender_name: Mapped[str]
    sender_role: Mapped[str]
    sender_client_id: Mapped[Optional[str]]


class RunModel(MyAbstractBase):
    __tablename__ = "training_runs"

    session_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sessions.id"))
    model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    completed_at_model_id: Mapped[Optional[str]] = mapped_column(ForeignKey("models.id"))
    round_timeout: Mapped[int]
    rounds: Mapped[Optional[int]]
    completed_at: Mapped[Optional[datetime]]


class TelemetryModel(MyAbstractBase):
    __tablename__ = "telemetry"

    key: Mapped[str] = mapped_column(String(255))
    value: Mapped[float]

    timestamp: Mapped[Optional[datetime]]

    sender_name: Mapped[str]
    sender_role: Mapped[str]
    sender_client_id: Mapped[Optional[str]]
