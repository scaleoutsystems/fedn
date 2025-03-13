from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class ClientDTO(BaseDTO):
    """Client data transfer object."""

    client_id: Optional[str] = Field(None)
    name: str = Field(None)
    combiner: str = Field(None)
    combiner_preferred: Optional[str] = Field(None)
    ip: str = Field(None)
    status: str = Field(None)
    last_seen: datetime = Field(None)
    package: Optional[str] = Field(None)
