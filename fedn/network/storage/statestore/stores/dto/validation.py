from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import AgentDTO, BaseDTO, Field, OptionalField


class ValidationDTO(BaseDTO):
    """Validation data transfer object."""

    validation_id: Optional[str] = OptionalField(None)
    model_id: Optional[str] = Field(None)
    data: Optional[str] = Field(None)
    correlation_id: Optional[str] = Field(None)
    timestamp: Optional[datetime] = Field(None)
    session_id: Optional[str] = Field(None)
    meta: Optional[str] = Field(None)
    sender: Optional[AgentDTO] = AgentDTO()
    receiver: Optional[AgentDTO] = AgentDTO()
