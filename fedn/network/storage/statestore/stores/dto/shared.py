"""Module contains minimal funcitonality mirrored from Pydantic BaseModel."""

from datetime import datetime
from typing import Any, Dict


class Field:
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        self.default_value = default_value


class OptionalField(Field):
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        super().__init__(default_value)


class BaseDTO:
    """BaseDTO for Data Transfer Objects."""

    committed_at: datetime = Field(None)

    def __init__(self, **kwargs) -> None:
        """Initialize BaseModel."""
        self.patch(kwargs)

    def __getattribute__(self, name: str):
        """Get attribute with default value if Field."""
        result = super().__getattribute__(name)
        if isinstance(result, Field):
            return result.default_value
        return result

    def model_dump(self, exclude_unset: bool = False) -> Dict[str, Any]:
        result = {}
        for k in self.__class__.get_all_fieldnames():
            v = super().__getattribute__(k)
            if isinstance(v, Field):
                if not exclude_unset:
                    result[k] = v.default_value
            else:
                result[k] = v
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation of BaseModel. (Overridable)

        This method is used to convert the BaseModel to a dictionary representation.
        """
        return self.model_dump(exclude_unset=False)

    def to_db(self, exclude_unset: bool = True) -> Dict[str, Any]:
        """Return dict representation of BaseModel for database storage."""
        return self.model_dump(exclude_unset=exclude_unset)

    def patch(self, value_dict: Dict[str, Any], throw_on_extra_keys: bool = True) -> "BaseDTO":
        """Patch BaseModel with value_dict."""
        for key, value in value_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif throw_on_extra_keys:
                raise ValueError(f"Invalid key: {key}")

        return self

    def populate_with(self, value_dict: Dict[str, Any]) -> "BaseDTO":
        """Populate BaseModel from value_dict.

        This method is used to apply a dictionary to the model ignoring any extra keys but requires all fields to be present.
        """
        for k in self.__class__.get_all_fieldnames():
            if k in value_dict:
                setattr(self, k, value_dict[k])
            elif isinstance(super().__getattribute__(k), OptionalField) or not isinstance(super().__getattribute__(k), Field) or k == "committed_at":
                pass
            else:
                raise ValueError(f"Missing key: {k}")
        return self

    @classmethod
    def get_all_fieldnames(cls) -> Dict[str, Any]:
        """Get all fields of the class and its superclasses."""
        keys = []
        for base in cls.__mro__:
            if hasattr(base, "__dict__"):
                for key, value in base.__dict__.items():
                    if isinstance(value, Field):
                        keys.append(key)
        return keys
