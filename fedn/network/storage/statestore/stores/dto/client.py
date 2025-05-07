from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID


class ClientDTO(BaseDTO):
    """Client data transfer object."""

    client_id: Optional[str] = PrimaryID(None)
    name: str = Field(None)
    combiner: str = Field(None)
    combiner_preferred: Optional[str] = Field(None)
    ip: Optional[str] = Field(None)
    status: str = Field(None)
    last_seen: datetime = Field(None)
    package: Optional[str] = Field(None)
