from enum import Enum


class SortOrder(Enum):
    ASCENDING = "ASC"
    DESCENDING = "DESC"


class EntityNotFound(Exception):
    pass


class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        super().__init__("Validation error on field {}: {}".format(field, message))
        self.field = field
        self.validation_message = message

    def user_message(self):
        return "Validation failed for field '{}' with message: {}".format(self.field, self.validation_message)


class MissingFieldError(Exception):
    def __init__(self, field: str, class_name: str):
        super().__init__("Missing field {} in class {}".format(field, class_name))
        self.field = field
        self.class_name = class_name

    def user_message(self):
        return "Class '{}' is missing field {}".format(self.class_name, self.field)
