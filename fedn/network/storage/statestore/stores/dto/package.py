from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class PackageDTO(BaseDTO):
    """Package data transfer object."""

    package_id: Optional[str] = Field(None)
    description: str = Field(None)
    file_name: str = Field(None)
    helper: str = Field(None)
    name: str = Field(None)
    storage_file_name: Optional[str] = Field(None)
    active: Optional[bool] = Field(False)


# def validate(item: Dict) -> Tuple[bool, str]:
#     if "file_name" not in item or not item["file_name"]:
#         return False, "File name is required"

#     if not allowed_file_extension(item["file_name"]):
#         return False, "File extension not allowed"

#     if "helper" not in item or not validate_helper(item["helper"]):
#         return False, "Helper is required"

#     return True, ""
