"""Module contains minimal funcitonality mirrored from Pydantic BaseModel."""

import copy
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class Field:
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        self.default_value = default_value

    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self.default_value))


class DTO(ABC):
    @abstractmethod
    def is_modified(self) -> bool:
        """Check if DTO has any modifications."""
        pass

    @abstractmethod
    def model_dump(self, exclude_unset: bool = False) -> Dict[str, Any]:
        """Dump DTO to dict."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation of DTO.

        This method is used to convert the DTO to a dictionary representation.
        """
        return self.model_dump(exclude_unset=False)

    def to_db(self, exclude_unset: bool = False) -> Dict[str, Any]:
        """Return dict representation of DTO for database storage."""
        return self.model_dump(exclude_unset=exclude_unset)

    @abstractmethod
    def verify(self):
        """Verify DTO and raise exception a if required field is missing."""
        pass

    @abstractmethod
    def patch_with(self, object: "DTO", throw_on_extra_keys: bool = True, verify: bool = False) -> "DTO":
        """Patch DTO with object.

        This method is used to apply a dict/list/DTO to the model with the option to throw an error if extra keys are present.
        It operates recursively if an dict/list are nested and matching fields are of type DTO.
        If verify is set to True, the DTO will be verified after the patch.
        """
        pass

    @abstractmethod
    def clear(self):
        """Reset DTO to initial values."""
        pass


class DictDTO(DTO):
    def __init__(self, **kwargs) -> None:
        """Initialize DTO.

        This method copies all class fields and patches the instance with the provided kwargs.
        """
        for k in self.get_all_fieldnames():
            v = getattr(self.__class__, k).default_value
            super().__setattr__(k, copy.deepcopy(v))
        super().__setattr__("_modified_fields", set())
        self.patch_with(kwargs)

    def __setattr__(self, name: str, value):
        """Set attribute and store which fields are modified."""
        if not self._is_field(name):
            super().__setattr__(name, value)
            return

        if issubclass(self._get_field_type(name), DTO):
            current_value = getattr(self, name)
            if value is None:
                current_value.clear()
            elif (
                issubclass(self._get_field_type(name), DictDTO)
                and not isinstance(value, (dict, DictDTO))
                or issubclass(self._get_field_type(name), ListDTO)
                and not isinstance(value, (list, ListDTO))
            ):
                raise ValueError(f"Can not set key: {name} to type {value.__class__.__name__} in {self.__class__.__name__}")
            else:
                current_value.clear()
                current_value.patch_with(value)
                self._modified_fields.add(name)
        else:
            super().__setattr__(name, value)
            self._modified_fields.add(name)

    def clear_field(self, fieldname):
        """Clear a field of modified value."""
        v = getattr(self.__class__, fieldname).default_value
        super().__setattr__(fieldname, copy.deepcopy(v))
        self._modified_fields.discard(fieldname)

    def clear(self):
        """Clear all modified fields."""
        for field_name in self.get_all_fieldnames():
            self.clear_field(field_name)

    def model_dump(self, exclude_unset: bool = False) -> Dict[str, Any]:
        result = {}
        for field_name in self.get_all_fieldnames():
            field_value = getattr(self, field_name)
            if exclude_unset and not self._is_field_modified(field_name):
                continue
            if isinstance(field_value, DTO):
                if not self._is_field_optional(field_name) or field_value.is_modified():
                    result[field_name] = field_value.model_dump(exclude_unset)
                else:
                    result[field_name] = None
            else:
                result[field_name] = field_value
        return result

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_unset=False)

    def to_db(self, exclude_unset: bool = False) -> Dict[str, Any]:
        return self.model_dump(exclude_unset=exclude_unset)

    def patch_with(self, value_dict: Union[Dict[str, Any], "DictDTO"], throw_on_extra_keys: bool = True, verify: bool = False) -> "DictDTO":
        if isinstance(value_dict, DictDTO):
            value_dict = value_dict.model_dump(exclude_unset=True)

        for key, value in value_dict.items():
            if self._is_field(key):
                setattr(self, key, value)
            elif throw_on_extra_keys:
                raise ValueError(f"Invalid key: {key} for {self.__class__.__name__}")

        if verify:
            self.verify()

        return self

    def populate_with(self, value_dict: Union[Dict[str, Any], "DictDTO"], throw_on_extra_keys: bool = True) -> "DictDTO":
        """Populate DTO from value_dict.

        This method is used to apply a dictionary to the model and requires all non-optional/previously unset fields to be present.
        It operates recursively if value_dict contains nested dictionaries and matching fields are of type DTO.

        A DTO populated with this method is guaranteed to have all required fields set.
        """
        return self.patch_with(value_dict, throw_on_extra_keys, verify=True)

    def is_modified(self):
        if bool(self._modified_fields):
            return True
        return any(isinstance(getattr(self, field_name), DTO) and getattr(self, field_name).is_modified() for field_name in self.get_all_fieldnames())

    def verify(self):
        for field_name in self.get_all_fieldnames():
            field_value = getattr(self, field_name)
            if isinstance(field_value, DTO):
                if not self._is_field_optional(field_name) or field_value.is_modified():
                    field_value.verify()
            elif not self._is_field_optional(field_name) and not self._is_field_modified(field_name):
                raise ValueError(f"Missing key: {field_name} for {self.__class__.__name__}")

    @classmethod
    def get_all_fieldnames(cls) -> Dict[str, Any]:
        """Get all fields of the class and its superclasses."""
        keys = []
        for base in cls.__mro__:
            if hasattr(base, "__dict__"):
                for key in base.__dict__.keys():
                    if cls._is_field(key):
                        keys.append(key)
        return keys

    # Private methods
    def _is_field_modified(self, field_name):
        """Check if a field is modified."""
        if field_name in self._modified_fields:
            return True
        field_value = getattr(self, field_name)
        if isinstance(field_value, DTO):
            return field_value.is_modified()
        return False

    @classmethod
    def _is_field(cls, field_name: str) -> bool:
        """Check if a field is a DTO."""
        return hasattr(cls, field_name) and isinstance(getattr(cls, field_name), Field)

    @classmethod
    def _is_field_optional(cls, field_name: str) -> bool:
        """Check if a field is optional."""
        for base in cls.__mro__:
            if hasattr(base, "__annotations__") and field_name in base.__annotations__:
                if cls._is_field(field_name):
                    return _is_optional(base.__annotations__[field_name])
        raise ValueError(f"Field {field_name} not found in {cls.__name__}")

    @classmethod
    def _get_field_type(cls, field_name):
        """Get the type of a field."""
        for base in cls.__mro__:
            if hasattr(base, "__annotations__") and field_name in base.__annotations__:
                return _get_type(base.__annotations__[field_name])
        raise ValueError(f"Field {field_name} not found in {cls.__name__}")

    def __deepcopy__(self, memo):
        return self.__class__(**copy.deepcopy(self.model_dump(exclude_unset=True)))


class BaseDTO(DictDTO):
    """BaseDTO for Data Transfer Objects."""

    committed_at: datetime = Field(None)

    def _is_field_optional(self, key):
        return super()._is_field_optional(key) or key == "committed_at"


class ListDTO(DTO, Generic[T]):
    """ListDTO for Data Transfer Objects."""

    items: List[T]

    def __init__(self, ListClass, *values) -> None:
        """Initialize ListDTO."""
        super().__init__()
        self._ListClass = ListClass
        self.items = []
        self._modified = False
        if values:
            self.patch_with(values)

    def model_dump(self, exclude_unset=False):
        if issubclass(self._ListClass, DTO):
            return [item.model_dump(exclude_unset) for item in self.items]
        return self.items

    def patch_with(self, value_list: Union[List[T], "ListDTO"], throw_on_extra_keys: bool = True, verify: bool = False) -> "ListDTO":
        """Patch ListDTO with list."""
        if isinstance(value_list, ListDTO):
            value_list = value_list.model_dump(exclude_unset=True)

        self.items.clear()
        self._modified = True

        for i, item in enumerate(value_list):
            if issubclass(self._ListClass, DTO):
                dto = self._ListClass()
                dto.patch_with(item, throw_on_extra_keys)
                self.items.append(dto)
            else:
                self.items.append(item)

        if verify:
            self.verify()

        return self

    def verify(self):
        if issubclass(self._ListClass, DTO):
            for item in self.items:
                item.verify()

    def clear(self):
        self.items.clear()
        self._modified = False

    def is_modified(self) -> bool:
        return self._modified or any(item.is_modified() for item in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key) -> T:
        return self.items[key]

    def __iter__(self):
        return iter(self.items)

    def append(self, item: T):
        if issubclass(self._ListClass, DTO):
            item = self._ListClass().patch_with(item)
            self.items.append(item)
        else:
            self.items.append(item)
        self._modified = True

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        return self.__class__(self._ListClass, *copy.deepcopy(self.model_dump(exclude_unset=True)))


class AgentDTO(DictDTO):
    """Agent data transfer object."""

    name: str = Field(None)
    role: str = Field(None)


# Private functions


def _is_optional(annotation) -> bool:
    """Check if a type annotation is Optional."""
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _get_type(annotation) -> type:
    """Get the type of a type annotation. If the annotation is Optional, return the inner type."""
    if get_origin(annotation) is Union:
        return get_args(annotation)[0]
    elif get_origin(annotation) is not None:
        return get_origin(annotation)
    return annotation
