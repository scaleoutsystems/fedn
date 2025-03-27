class EntityNotFound(Exception):
    pass


class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        super().__init__("Validation error on field {}: {}".format(field, message))
        self.field = field
        self.validation_message = message


class MissingFieldError(Exception):
    def __init__(self, field: str, class_name: str):
        super().__init__("Missing field {} in class {}".format(field, class_name))
        self.field = field
        self.class_name = class_name
