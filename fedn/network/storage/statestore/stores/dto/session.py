from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, DictDTO, Field


class SessionConfigDTO(DictDTO):
    """SessionConfig data transfer object."""

    aggregator: str = Field(None)
    aggregator_kwargs: Optional[str] = Field(None)
    round_timeout: int = Field(None)
    buffer_size: int = Field(None)
    rounds: Optional[int] = Field(None)
    delete_models_storage: bool = Field(None)
    clients_required: int = Field(None)
    requested_clients: Optional[int] = Field(None)
    validate: bool = Field(None)
    helper_type: str = Field(None)
    model_id: str = Field(None)
    server_functions: Optional[str] = Field(None)


class SessionDTO(BaseDTO):
    """Session data transfer object."""

    session_id: str = Field(None)
    name: str = Field(None)
    status: Optional[str] = Field("Created")
    session_config: SessionConfigDTO = Field(SessionConfigDTO())
