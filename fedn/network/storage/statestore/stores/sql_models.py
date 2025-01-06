from typing import List, Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fedn.network.storage.statestore.stores.store import MyAbstractBase


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
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    session: Mapped["SessionModel"] = relationship(back_populates="session_config")


class SessionModel(MyAbstractBase):
    __tablename__ = "sessions"

    name: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255))
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
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    session: Mapped[Optional["SessionModel"]] = relationship(back_populates="round_configs")
    round: Mapped["RoundModel"] = relationship(back_populates="round_config")
    task: Mapped[str] = mapped_column(String(255))


class RoundDataModel(MyAbstractBase):
    __tablename__ = "round_data"

    time_commit: Mapped[float]
    reduce_time_aggregate_models: Mapped[float]
    reduce_time_fetch_models: Mapped[float]
    reduce_time_load_model: Mapped[float]
    round: Mapped["RoundModel"] = relationship(back_populates="round_data")


class RoundCombinerModel(MyAbstractBase):
    __tablename__ = "round_combiners"

    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    name: Mapped[str] = mapped_column(String(255))
    round_id: Mapped[str]
    parent_round_id: Mapped[str] = mapped_column(ForeignKey("rounds.id"))
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

    data_aggregation_time_nr_aggregated_models: Mapped[int]
    data_aggregation_time_time_model_aggregation: Mapped[float]
    data_aggregation_time_time_model_load: Mapped[float]
    data_nr_expected_updates: Mapped[int]
    data_nr_required_updates: Mapped[int]
    data_time_combination: Mapped[float]
    data_timeout: Mapped[float]


class RoundModel(MyAbstractBase):
    __tablename__ = "rounds"

    round_id: Mapped[str] = mapped_column(unique=True)  # TODO: Add unique constraint. Does this work?
    status: Mapped[str] = mapped_column()
    round_config: Mapped["RoundConfigModel"] = relationship(back_populates="round")
    round_data: Mapped["RoundDataModel"] = relationship(back_populates="round")
    combiners: Mapped[List["RoundCombinerModel"]] = relationship(back_populates="round")
