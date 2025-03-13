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
