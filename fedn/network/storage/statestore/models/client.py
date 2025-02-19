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

    @staticmethod
    def from_dict(value_dict: Dict[str, Any], throw_on_extra_keys: bool = True):
        client = Client()
        client.patch(value_dict, throw_on_extra_keys)
        return client

    def to_dict(self):
        return self.model_dump(exclude_unset=True)

    def patch(self, value_dict: Dict[str, Any], throw_on_extra_keys: bool = True):
        for key, value in value_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif throw_on_extra_keys:
                raise ValueError(f"Invalid key: {key}")
