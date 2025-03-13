from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import AgentDTO, BaseDTO, Field


class StatusDTO(BaseDTO):
    """Status data transfer object."""

    status_id: Optional[str] = Field(None)
    status: Optional[str] = Field(None)
    timestamp: Optional[datetime] = Field(None)
    log_level: Optional[str] = Field(None)
    data: Optional[str] = Field(None)
    correlation_id: Optional[str] = Field(None)
    type: Optional[str] = Field(None)
    extra: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)
    sender: Optional[AgentDTO] = Field(AgentDTO())
