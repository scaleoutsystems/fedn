from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID


class ModelDTO(BaseDTO):
    """Model data transfer object."""

    model_id: Optional[str] = PrimaryID(None)
    name: str = Field(None)
    parent_model: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)

    def to_dict(self):
        res = super().to_dict()
        # TODO: Remove this when we have migrated all model to model_id
        res["model"] = self.model_id
        return res

    def patch_with(self, value_dict, throw_on_extra_keys=True, verify=False):
        # TODO: Remove this when we have migrated all model to model_id
        if "model" in value_dict:
            if "model_id" in value_dict and value_dict["model_id"] != value_dict["model"]:
                raise ValueError("Cannot set both model and model_id in ModelDTO")
            value_dict["model_id"] = value_dict.pop("model")
        return super().patch_with(value_dict, throw_on_extra_keys)
