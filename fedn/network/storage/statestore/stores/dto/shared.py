"""Module contains minimal funcitonality mirrored from Pydantic BaseModel."""

import copy
from datetime import datetime
from typing import Any, Dict, Union, get_args, get_origin


class Field:
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        self.default_value = default_value

    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self.default_value))


def _is_optional(annotation) -> bool:
    """Check if a type annotation is Optional."""
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


class DTO:
    def __init__(self, **kwargs) -> None:
        """Initialize BaseModel.

        This method copies all class fields and patches the instance with the provided kwargs.
        """
        for k in self.get_all_fieldnames():
            v = getattr(self.__class__, k).default_value
            super().__setattr__(k, copy.deepcopy(v))
        super().__setattr__("_modified_fields", set())
        self.patch(kwargs)

    def __setattr__(self, name: str, value):
        """Set attribute and store which fields are modified."""
        super().__setattr__(name, value)
        if not isinstance(value, DTO):
            self._modified_fields.add(name)

    def clear_field(self, fieldname):
        """Clear a field of modified value."""
        v = getattr(self.__class__, fieldname).default_value
        super().__setattr__(fieldname, copy.deepcopy(v))
        self._modified_fields.remove(fieldname)

    def clear_all_fields(self):
        """Clear all modified fields."""
        for fieldname in self._modified_fields:
            self.clear_field(fieldname)

    def model_dump(self, exclude_unset: bool = False) -> Dict[str, Any]:
        """Dump BaseModel to dict."""
        result = {}
        for field_name in self.get_all_fieldnames():
            field_value = getattr(self, field_name)
            if exclude_unset and not self._is_field_modified(field_name):
                continue
            if isinstance(field_value, DTO):
                result[field_name] = field_value.model_dump(exclude_unset)
            else:
                result[field_name] = field_value
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation of BaseModel.

        This method is used to convert the BaseModel to a dictionary representation.
        """
        return self.model_dump(exclude_unset=False)

    def to_db(self, exclude_unset: bool = False) -> Dict[str, Any]:
        """Return dict representation of BaseModel for database storage."""
        return self.model_dump(exclude_unset=exclude_unset)

    def patch(self, value_dict: Union[Dict[str, Any], "DTO"], throw_on_extra_keys: bool = True) -> "DTO":
        """Patch BaseModel with value_dict.

        This method is used to apply a dictionary to the model with the option to throw an error if extra keys are present.
        It operates recursively if value_dict contains nested dictionaries and matching fields are of type BaseModel.
        """
        if isinstance(value_dict, DTO):
            value_dict = value_dict.model_dump(exclude_unset=True)

        for key, value in value_dict.items():
            if self._valid_fieldname(key):
                if isinstance(getattr(self, key), DTO):
                    if isinstance(value_dict[key], dict):
                        getattr(self, key).patch(value_dict[key], throw_on_extra_keys)
                    elif isinstance(value_dict[key], DTO) and getattr(self, key).__class__ == value_dict[key].__class__:
                        setattr(self, key, value_dict[key])
                    else:
                        raise ValueError(f"Can not set key: {key} to type {value.__class__.__name__} in {self.__class__.__name__}")
                else:
                    setattr(self, key, value)
            elif throw_on_extra_keys:
                raise ValueError(f"Invalid key: {key} for {self.__class__.__name__}")

        return self

    def populate(self, **kwargs) -> "DTO":
        """Populate BaseModel with kwargs."""
        return self.populate_with(kwargs)

    def populate_with(self, value_dict: Union[Dict[str, Any], "DTO"], throw_on_extra_keys: bool = True) -> "DTO":
        """Populate BaseModel from value_dict.

        This method is used to apply a dictionary to the model and requires all non-optional/previously unset fields to be present.
        It operates recursively if value_dict contains nested dictionaries and matching fields are of type BaseModel.

        A DTO populated with this method is guaranteed to have all required fields set.
        """
        if isinstance(value_dict, DTO):
            value_dict = value_dict.model_dump(exclude_unset=True)
        else:
            # Make a copy of the value_dict to avoid modifying the original
            value_dict = copy.copy(value_dict)

        for field_name in self.get_all_fieldnames():
            field_value = getattr(self, field_name)
            if field_name in value_dict:
                if isinstance(field_value, DTO):
                    if isinstance(value_dict[field_name], dict):
                        field_value.populate_with(value_dict[field_name], throw_on_extra_keys=throw_on_extra_keys)
                    elif isinstance(value_dict[field_name], DTO):
                        field_value.populate_with(value_dict[field_name].model_dump(exclude_unset=True), throw_on_extra_keys=throw_on_extra_keys)
                    else:
                        raise ValueError(f"Can not set key: {field_name} to type {value_dict[field_name].__class__.__name__} in {self.__class__.__name__}")
                else:
                    setattr(self, field_name, value_dict[field_name])
                del value_dict[field_name]
            elif isinstance(field_value, DTO):
                if self._is_field_optional(field_name) and not field_value.is_modified():
                    pass
                else:
                    # Child DTOs might already be (partial) populated, populate them with empty dict to check for missing keys
                    field_value.populate_with({})
            elif self._is_field_optional(field_name):
                pass
            elif not self._is_field_modified(field_name):
                raise ValueError(f"Missing key: {field_name} for {self.__class__.__name__}")
            else:
                pass  # Field already set

        if value_dict and throw_on_extra_keys:
            raise ValueError(f"Invalid key(s): {list(value_dict.keys())} for {self.__class__.__name__}")

        return self

    def is_populated(self) -> bool:
        """Check if DTO is fully populated."""
        for field_name in self.get_all_fieldnames():
            field_value = getattr(self, field_name)
            if isinstance(field_value, DTO):
                if not (field_value.is_populated() or (self._is_field_optional(field_name) and not field_value.is_modified())):
                    return False
            elif not (self._is_field_modified(field_name) or self._is_field_optional(field_name)):
                return False
        return True

    def is_modified(self):
        """Check if DTO has any modifications."""
        if bool(self._modified_fields):
            return True
        return any(isinstance(getattr(self, field_name), DTO) and getattr(self, field_name).is_modified() for field_name in self.get_all_fieldnames())

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

    # Private methods

    def _is_field_optional(self, key: str) -> bool:
        """Check if a field is optional."""
        for base in self.__class__.__mro__:
            if hasattr(base, "__annotations__") and isinstance(getattr(base, key), Field) and key in base.__annotations__:
                return _is_optional(base.__annotations__[key])

    def _valid_fieldname(self, field_name):
        return field_name in self.get_all_fieldnames()

    def _is_field_modified(self, field_name):
        """Check if a field is modified."""
        if field_name in self._modified_fields:
            return True
        if isinstance(getattr(self, field_name), DTO):
            return getattr(self, field_name).is_modified()
        return False

    def __deepcopy__(self, memo):
        return self.__class__(**copy.deepcopy(self.model_dump(exclude_unset=True)))


class BaseDTO(DTO):
    """BaseDTO for Data Transfer Objects."""

    committed_at: datetime = Field(None)

    def _is_field_optional(self, key):
        return super()._is_field_optional(key) or key == "committed_at"


class AgentDTO(DTO):
    """Agent data transfer object."""

    name: str = Field(None)
    role: str = Field(None)
