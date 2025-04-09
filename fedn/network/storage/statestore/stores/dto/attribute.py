from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, NodeDTO, PrimaryID


class AttributeDTO(BaseDTO):
    attribute_id: str = PrimaryID(None)

    key: str = Field(None)
    value: str = Field(None)

    timestamp: Optional[datetime] = Field(None)

    sender: NodeDTO = Field(NodeDTO())
