from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseModel, Field


class Client(BaseModel):
    """Client data transfer object."""

    client_id: Optional[str] = Field(None)
    name: str = Field(None)
    combiner: str = Field(None)
    combiner_preferred: str = Field(None)
    ip: str = Field(None)
    status: str = Field(None)
    last_seen: datetime = Field(None)
    package: Optional[str] = Field(None)
