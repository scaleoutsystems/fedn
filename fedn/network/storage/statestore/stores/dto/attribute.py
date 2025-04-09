from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, NodeDTO, PrimaryID


class AttributeDTO(BaseDTO):
    metric_id: str = PrimaryID(None)

    attribute: dict = Field({})

    timestamp: Optional[datetime] = Field(None)

    sender: NodeDTO = Field(NodeDTO())
