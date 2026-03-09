from enum import Enum
from typing import Union


class StatusType(Enum):
    EMPTY = ""
    COMPLETED = "COMPLETED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    NEW = "NEW"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def matches(cls, status1: Union["StatusType", str], status2: Union["StatusType", str]) -> bool:
        """Check if two status types match, regardless of whether they are strings or StatusType enums."""
        if isinstance(status1, str):
            status1 = cls.from_string(status1)
        if isinstance(status2, str):
            status2 = cls.from_string(status2)
        return status1 == status2

    @classmethod
    def from_string(cls, status_str: str) -> "StatusType":
        """Convert a string to a StatusType enum."""
        if status_str in cls._value2member_map_:
            return cls(status_str)
        elif status_str == "EMPTY" or status_str is None:
            return cls.EMPTY
        else:
            raise ValueError(f"Unknown status string: {status_str}")
