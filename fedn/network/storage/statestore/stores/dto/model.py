from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class ModelDTO(BaseDTO):
    """Model data transfer object."""

    model_id: Optional[str] = Field(None)
    name: str = Field(None)
    parent_model: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)

    def to_dict(self, exclude_unset=True):
        res = super().to_dict(exclude_unset)
        res["model"] = self.model_id
        return res

    def patch(self, value_dict, throw_on_extra_keys=True):
        if "model" in value_dict:
            value_dict["model_id"] = value_dict.pop("model")
        return super().patch(value_dict, throw_on_extra_keys)
