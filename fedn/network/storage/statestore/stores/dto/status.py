from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, NodeDTO, PrimaryID


class StatusDTO(BaseDTO):
    """Status data transfer object."""

    # TODO: Correct which fields are optional
    status_id: Optional[str] = PrimaryID(None)
    status: Optional[str] = Field(None)
    timestamp: Optional[datetime] = Field(None)
    log_level: Optional[str] = Field(None)
    data: Optional[str] = Field(None)
    correlation_id: Optional[str] = Field(None)
    type: Optional[str] = Field(None)
    extra: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)
    sender: Optional[NodeDTO] = Field(NodeDTO())
