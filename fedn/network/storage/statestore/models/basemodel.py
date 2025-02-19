"""Module contains minimal funcitonality mirrored from Pydantic BaseModel."""


class Field:
    def __init__(self, default_value):
        self.default_value = default_value


class BaseModel:
    def __init__(self):
        pass

    def model_dump(self, exclude_unset: bool = False):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Field):
                if not exclude_unset:
                    result[k] = v.default_value
            else:
                result[k] = v
        return result
