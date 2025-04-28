from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, NodeDTO, PrimaryID


class TelemetryDTO(BaseDTO):
    telemetry_id: str = PrimaryID(None)

    key: str = Field(None)
    value: float = Field(None)

    timestamp: Optional[datetime] = Field(None)

    sender: NodeDTO = Field(NodeDTO())
