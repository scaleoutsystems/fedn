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
