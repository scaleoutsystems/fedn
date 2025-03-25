from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import AgentDTO, BaseDTO, Field, PrimaryID


class ValidationDTO(BaseDTO):
    """Validation data transfer object."""

    # TODO: Correct which fields are optional
    validation_id: Optional[str] = PrimaryID(None)
    model_id: Optional[str] = Field(None)
    data: Optional[str] = Field(None)
    correlation_id: Optional[str] = Field(None)
    timestamp: Optional[datetime] = Field(None)
    session_id: Optional[str] = Field(None)
    meta: Optional[str] = Field(None)
    sender: Optional[AgentDTO] = Field(AgentDTO())
    receiver: Optional[AgentDTO] = Field(AgentDTO())
