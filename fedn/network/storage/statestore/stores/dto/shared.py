"""Module contains minimal funcitonality mirrored from Pydantic BaseModel."""

import copy
from datetime import datetime
from typing import Any, Dict


class Field:
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        self.default_value = default_value

    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self.default_value))


class OptionalField(Field):
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        super().__init__(default_value)


class DTO:
    def __init__(self, **kwargs) -> None:
        """Initialize BaseModel.

        This method copieas all class fields and patches the instance with the provided kwargs.
        """
        for k in self.__class__.get_all_fieldnames():
            v = super().__getattribute__(k)
            setattr(self, k, copy.deepcopy(v))
        self.patch(kwargs)

    # use super().__getattribute__ to avoid recursion
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
            elif isinstance(v, DTO):
                sub_v = v.model_dump(exclude_unset)
                if sub_v:
                    result[k] = sub_v
            else:
                result[k] = v
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation of BaseModel. (Overridable)

        This method is used to convert the BaseModel to a dictionary representation.
        """
        return self.model_dump(exclude_unset=False)

    def to_db(self, exclude_unset: bool = False) -> Dict[str, Any]:
        """Return dict representation of BaseModel for database storage."""
        return self.model_dump(exclude_unset=exclude_unset)

    def patch(self, value_dict: Dict[str, Any], throw_on_extra_keys: bool = True) -> "DTO":
        """Patch BaseModel with value_dict.

        This method is used to apply a dictionary to the model with the option to throw an error if extra keys are present.
        It operates recursively if value_dict contains nested dictionaries and matching fields are of type BaseModel.
        """
        for key, value in value_dict.items():
            if hasattr(self, key):
                if isinstance(super().__getattribute__(key), DTO):
                    if isinstance(value_dict[key], dict):
                        getattr(self, key).patch(value_dict[key], throw_on_extra_keys)
                    elif isinstance(value_dict[key], DTO):
                        setattr(self, key, value_dict[key])
                    else:
                        raise ValueError(f"Invalid value for key: {key}")
                else:
                    setattr(self, key, value)
            elif throw_on_extra_keys:
                raise ValueError(f"Invalid key: {key}")

        return self

    def populate_with(self, value_dict: Dict[str, Any]) -> "DTO":
        """Populate BaseModel from value_dict.

        This method is used to apply a dictionary to the model ignoring any extra keys but requires all fields to be present.
        It operates recursively if value_dict contains nested dictionaries and matching fields are of type BaseModel.
        """
        for k in self.__class__.get_all_fieldnames():
            if k in value_dict:
                if isinstance(super().__getattribute__(k), DTO):
                    if isinstance(value_dict[k], dict):
                        getattr(self, k).populate_with(value_dict[k])
                    elif isinstance(value_dict[k], DTO):
                        setattr(self, k, value_dict[k])
                    else:
                        raise ValueError(f"Invalid value for key: {k}")
                else:
                    setattr(self, k, value_dict[k])
            elif isinstance(super().__getattribute__(k), OptionalField) or k == "committed_at":
                pass
            elif isinstance(super().__getattribute__(k), DTO):
                getattr(self, k).populate_with({})
            elif isinstance(super().__getattribute__(k), Field):
                raise ValueError(f"Missing key: {k}")
            else:
                pass  # Field already set

        return self

    @classmethod
    def get_all_fieldnames(cls) -> Dict[str, Any]:
        """Get all fields of the class and its superclasses."""
        keys = []
        for base in cls.__mro__:
            if hasattr(base, "__dict__"):
                for key, value in base.__dict__.items():
                    if isinstance(value, (Field, DTO)):
                        keys.append(key)
        return keys

    def __deepcopy__(self, memo):
        return self.__class__(**copy.deepcopy(self.model_dump(exclude_unset=True)))


class BaseDTO(DTO):
    """BaseDTO for Data Transfer Objects."""

    committed_at: datetime = Field(None)


class AgentDTO(DTO):
    """Agent data transfer object."""

    name: str = Field(None)
    role: str = Field(None)
