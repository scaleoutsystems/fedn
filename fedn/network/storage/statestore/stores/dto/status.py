from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import AgentDTO, BaseDTO, OptionalField


class StatusDTO(BaseDTO):
    """Status data transfer object."""

    status_id: Optional[str] = OptionalField(None)
    status: Optional[str] = OptionalField(None)
    timestamp: Optional[datetime] = OptionalField(None)
    log_level: Optional[str] = OptionalField(None)
    data: Optional[str] = OptionalField(None)
    correlation_id: Optional[str] = OptionalField(None)
    type: Optional[str] = OptionalField(None)
    extra: Optional[str] = OptionalField(None)
    session_id: Optional[str] = OptionalField(None)
    sender: Optional[AgentDTO] = AgentDTO()
