from enum import Enum


class Role(Enum):
    WORKER = 1
    COMBINER = 2
    REDUCER = 3
    OTHER = 4
