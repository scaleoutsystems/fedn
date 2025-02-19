from datetime import datetime
from typing import Any, Dict, Optional

from .basemodel import BaseModel, Field


class Client(BaseModel):
    id: str = Field(None)
    name: str = Field(None)
    combiner: str = Field(None)
    combiner_preferred: str = Field(None)
    ip: str = Field(None)
    status: str = Field(None)
    last_seen: datetime = Field(None)
    client_id: Optional[str] = Field(None)
    package: Optional[str] = Field(None)
