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


"""
log_metric(loss: 1, accarcy: 0.9) # 0
log_metric(loss: 0.9) # 1
log_metric(accarcy: 0.9) # 2
log_metric(loss: 0.8, accarcy: 0.9) # 3

new round

log_metric(loss: 0.8, accarcy: 0.9) # 0"
"""
