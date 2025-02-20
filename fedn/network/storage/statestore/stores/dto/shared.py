"""Module contains minimal funcitonality mirrored from Pydantic BaseModel."""

from typing import Any, Dict


class Field:
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        self.default_value = default_value


class BaseModel:
    """BaseModel for Data Transfer Objects."""

    def __init__(self, **kwargs) -> None:
        """Initialize BaseModel."""
        if "id" in kwargs:
            del kwargs["id"]
        self.patch(kwargs)

    def __getattribute__(self, name: str):
        """Get attribute with default value if Field."""
        result = super().__getattribute__(name)
        if isinstance(result, Field):
            return result.default_value
        return result

    def model_dump(self, exclude_unset: bool = False) -> Dict[str, Any]:
        result = {}
        for k in self.__dict__:
            v = super().__getattribute__(k)
            if isinstance(v, Field):
                if not exclude_unset:
                    result[k] = v.default_value
            else:
                result[k] = v
        return result

    def to_dict(self, exclude_unset: bool = True) -> Dict[str, Any]:
        """Return dict representation of BaseModel.

        :param exclude_unset: Exclude unset values, defaults to True
        """
        return self.model_dump(exclude_unset=exclude_unset)

    def patch(self, value_dict: Dict[str, Any], throw_on_extra_keys: bool = True) -> None:
        """Patch BaseModel with value_dict."""
        for key, value in value_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif throw_on_extra_keys:
                raise ValueError(f"Invalid key: {key}")
