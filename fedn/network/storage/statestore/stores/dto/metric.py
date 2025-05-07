from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, NodeDTO, PrimaryID


class MetricDTO(BaseDTO):
    metric_id: str = PrimaryID(None)

    key: str = Field(None)
    value: float = Field(None)

    timestamp: Optional[datetime] = Field(None)

    sender: NodeDTO = Field(NodeDTO())

    model_id: str = Field(None)
    step: Optional[int] = Field(None)

    round_id: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)
