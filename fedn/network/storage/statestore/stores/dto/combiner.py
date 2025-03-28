from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID


class CombinerDTO(BaseDTO):
    """Client data transfer object."""

    combiner_id: Optional[str] = PrimaryID(None)
    name: str = Field(None)
    address: str = Field(None)
    fqdn: str = Field(None)
    ip: str = Field(None)
    parent: dict = Field(None)
    port: int = Field(None)
    updated_at: datetime = Field(None)
