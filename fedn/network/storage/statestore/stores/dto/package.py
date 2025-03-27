from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID, validator
from fedn.network.storage.statestore.stores.shared import ValidationError


def allowed_file_extension(filename: str, ALLOWED_EXTENSIONS={"gz", "bz2", "tar", "zip", "tgz"}) -> bool:
    """Check if file extension is allowed.

    :param filename: The filename to check.
    :type filename: str
    :return: True and extension str if file extension is allowed, else False and None.
    :rtype: Tuple (bool, str)
    """
    if "." in filename:
        extension = filename.rsplit(".", 1)[1].lower()
        if extension in ALLOWED_EXTENSIONS:
            return True

    return False


def validate_helper(helper: str) -> bool:
    if not helper or helper == "" or helper not in ["numpyhelper", "binaryhelper", "androidhelper"]:
        return False
    return True


class PackageDTO(BaseDTO):
    """Package data transfer object."""

    package_id: Optional[str] = PrimaryID(None)
    description: str = Field(None)
    file_name: str = Field(None)
    helper: str = Field(None)
    name: str = Field(None)
    storage_file_name: Optional[str] = Field(None)
    active: Optional[bool] = Field(False)

    @validator
    def validate(self):
        if not self.file_name:
            raise ValidationError("file_name", "File name is required")

        if not allowed_file_extension(self.file_name):
            return ValidationError("file_name", "File extension not allowed")

        if not self.helper or not validate_helper(self.helper):
            return ValidationError("helper", "Helper is required or is invalid")
