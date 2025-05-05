"""Module contains minimal funcitonality mirrored from Pydantic BaseModel."""

import copy
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, get_args, get_origin

from fedn.network.storage.statestore.stores.shared import MissingFieldError

T = TypeVar("T")


class Field:
    """Field class for DTOs."""

    def __init__(self, default_value) -> None:
        """Initialize Field with default value."""
        self.default_value = default_value

    def __deepcopy__(self, memo):
        return self.__class__(copy.deepcopy(self.default_value))


class validator:  # noqa: N801
    """decorator class for validation methods of DTOs."""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class DTO(ABC):
    @abstractmethod
    def has_modifications(self) -> bool:
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
    def check_validity(self):
        """Verify DTO and raise ValidationError a if required field is missing or invalid."""
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
    def clear_all_changes(self):
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
        self._modified_fields = set()
        self.patch_with(kwargs)

    def __setattr__(self, name: str, value):
        """Set attribute and store which fields are modified."""
        if not self._is_field(name):
            super().__setattr__(name, value)
            return

        if issubclass(self._get_field_type(name), DTO):
            if value is None:
                super().__setattr__(name, None)
            elif isinstance(value, self._get_field_type(name)):
                super().__setattr__(name, value)
            elif issubclass(self._get_field_type(name), DictDTO):
                if not isinstance(value, dict):
                    raise ValueError(f"Can not set key: {name} to type {value.__class__.__name__} in {self.__class__.__name__}")
                new_value: DictDTO = self._get_field_type(name)()
                new_value.patch_with(value)
                super().__setattr__(name, new_value)
            elif issubclass(self._get_field_type(name), ListDTO):
                if not isinstance(value, list):
                    raise ValueError(f"Can not set key: {name} to type {value.__class__.__name__} in {self.__class__.__name__}")
                new_value: ListDTO = ListDTO(self._get_list_type(name))
                new_value.patch_with(value)
                super().__setattr__(name, new_value)
            else:
                raise ValueError(f"Can not set key: {name} to type {value.__class__.__name__} in {self.__class__.__name__}")
        else:
            super().__setattr__(name, value)
        self._modified_fields.add(name)

    def clear_field(self, fieldname):
        """Clear a field of modified value."""
        v = getattr(self.__class__, fieldname).default_value
        super().__setattr__(fieldname, copy.deepcopy(v))
        self._modified_fields.discard(fieldname)

    def clear_all_changes(self):
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
                result[field_name] = field_value.model_dump(exclude_unset)
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
            self.check_validity()

        return self

    def populate_with(self, value_dict: Union[Dict[str, Any], "DictDTO"], throw_on_extra_keys: bool = True) -> "DictDTO":
        """Populate DTO from value_dict.

        This method is used to apply a dictionary to the model and requires all non-optional/previously unset fields to be present.
        It operates recursively if value_dict contains nested dictionaries and matching fields are of type DTO.

        A DTO populated with this method is guaranteed to have all required fields set.
        """
        return self.patch_with(value_dict, throw_on_extra_keys, verify=True)

    def has_modifications(self):
        if bool(self._modified_fields):
            return True
        return any(isinstance(getattr(self, field_name), DTO) and getattr(self, field_name).has_modifications() for field_name in self.get_all_fieldnames())

    def check_validity(self, exclude_primary_id=False):
        for field_name in self.get_all_fieldnames():
            field_value = getattr(self, field_name)
            if isinstance(field_value, DTO):
                if not self._is_field_optional(field_name) or field_value.has_modifications():
                    field_value.check_validity()
            elif not self._is_field_optional(field_name) and not self._is_field_modified(field_name):
                if not (self._is_primary_id(field_name) and exclude_primary_id):
                    raise MissingFieldError(field_name, self.__class__.__name__)
        self._run_validators()

    def _run_validators(self):
        for field_name in self.get_all_validators():
            validator = getattr(self, field_name)
            validator(self)

    @classmethod
    def get_all_fieldnames(cls) -> List[str]:
        """Get all fields of the class and its superclasses."""
        keys = []
        for base in cls.__mro__:
            if hasattr(base, "__dict__"):
                for key in base.__dict__.keys():
                    if cls._is_field(key):
                        keys.append(key)
        return keys

    @classmethod
    def get_all_validators(cls) -> List[str]:
        """Get all validators of the class and its superclasses."""
        keys = []
        for base in cls.__mro__:
            if hasattr(base, "__dict__"):
                for key in base.__dict__.keys():
                    if cls._is_validator(key):
                        keys.append(key)
        return keys

    # Private methods
    def _is_field_modified(self, field_name):
        """Check if a field is modified."""
        if field_name in self._modified_fields:
            return True
        field_value = getattr(self, field_name)
        if isinstance(field_value, DTO):
            return field_value.has_modifications()
        return False

    @classmethod
    def _is_field(cls, field_name: str) -> bool:
        """Check if a attribute is a Field."""
        return hasattr(cls, field_name) and isinstance(getattr(cls, field_name), Field)

    @classmethod
    def _is_primary_id(cls, field_name: str) -> bool:
        """Check if a attribute is a PrimaryID."""
        return hasattr(cls, field_name) and isinstance(getattr(cls, field_name), PrimaryID)

    @classmethod
    def _is_validator(cls, field_name: str) -> bool:
        """Check if a attribute is a validator."""
        return hasattr(cls, field_name) and isinstance(getattr(cls, field_name), validator)

    @classmethod
    def _is_field_optional(cls, field_name: str) -> bool:
        """Check if a field is optional."""
        for base in cls.__mro__:
            if hasattr(base, "__annotations__") and field_name in base.__annotations__:
                if cls._is_field(field_name):
                    return _is_optional(base.__annotations__[field_name])
        raise AttributeError(f"Field {field_name} not found in {cls.__name__}")

    @classmethod
    def _get_field_type(cls, field_name):
        """Get the type of a field."""
        for base in cls.__mro__:
            if hasattr(base, "__annotations__") and field_name in base.__annotations__:
                return _get_type(base.__annotations__[field_name])
        raise AttributeError(f"Field {field_name} not found in {cls.__name__}")

    @classmethod
    def _get_list_type(cls, field_name):
        """Get the type of a field."""
        for base in cls.__mro__:
            if hasattr(base, "__annotations__") and field_name in base.__annotations__:
                return _get_list_type(base.__annotations__[field_name])
        raise AttributeError(f"Field {field_name} not found in {cls.__name__}")

    def __deepcopy__(self, memo):
        return self.__class__(**copy.deepcopy(self.model_dump(exclude_unset=True)))


class ListDTO(DTO, Generic[T]):
    """ListDTO for Data Transfer Objects."""

    def __init__(self, ListClass, *values) -> None:
        """Initialize ListDTO."""
        super().__init__()
        self._ListClass = ListClass
        self.items: List[T] = []
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
            self.check_validity()

        return self

    def check_validity(self):
        if issubclass(self._ListClass, DTO):
            for item in self.items:
                item.check_validity()

    def clear_all_changes(self):
        self.items.clear()
        self._modified = False

    def has_modifications(self) -> bool:
        return self._modified or any(isinstance(item, DTO) and item.has_modifications() for item in self.items)

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


class PrimaryID(Field):
    """PrimaryID field for DTOs."""

    pass


class BaseDTO(DictDTO):
    """BaseDTO for Data Transfer Objects."""

    committed_at: datetime = Field(None)
    updated_at: datetime = Field(None)

    @property
    def primary_id(self) -> str:
        """Get the id of the DTO."""
        for base in self.__class__.__mro__:
            if hasattr(base, "__dict__"):
                for key in base.__dict__.keys():
                    if isinstance(getattr(self.__class__, key), PrimaryID):
                        return getattr(self, key)
        raise AttributeError(f"{self.__class__.__name__} has no field of type PrimaryID")

    def primary_key(self) -> str:
        """Get the key of the primary id."""
        for base in self.__class__.__mro__:
            if hasattr(base, "__dict__"):
                for key in base.__dict__.keys():
                    if isinstance(getattr(self.__class__, key), PrimaryID):
                        return key
        raise AttributeError(f"{self.__class__.__name__} has no field of type PrimaryID")

    def _is_field_optional(self, key):
        return super()._is_field_optional(key) or key in ["committed_at", "updated_at"]


class NodeDTO(DictDTO):
    """Agent data transfer object."""

    name: str = Field(None)
    role: str = Field(None)
    client_id: Optional[str] = Field(None)


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


def _get_list_type(annotation) -> type:
    """Get the type of a type annotation. If the annotation is Optional, return the inner type."""
    if get_origin(annotation) is ListDTO:
        return get_args(annotation)[0]
    else:
        return None
