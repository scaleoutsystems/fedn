from fedn.common.exceptions import InvalidParameterError


class Parameters(dict):
    """Represents a collection of parameters.

    Extends dict and adds functionality to validate
    paramteres types against a user-provided schema.

    Example of use:
        p = Parameters({'n_iter': 10, 'beta': 1e-2})
        p.validate({'n_iter': int, 'beta': float})

    """

    def __init__(self, parameters=None):
        """ """
        if parameters:
            for key, value in parameters.items():
                self.__setitem__(key, value)

    def validate(self, parameter_schema):
        """Validate parameters against a schema.

        :param parameter_schema: mapping of parameter name and data type.
        :type parameter_schema: dict
        :return: True if the parameters validate
        :rtype: bool
        """
        for key, value in self.items():
            if key not in parameter_schema.keys():
                raise InvalidParameterError("Parameter {} not in paramter schema".format(key))
            else:
                type = parameter_schema[key]
                self._validate_parameter_type(key, value, type)

        return True

    def _validate_parameter_type(self, key, value, type):
        """Validate that parameters values matches the data type.

        :param key: mapping of parameter name and data type.
        :type parameter_schema: dict
        :return: True if the parameters validate
        :rtype: bool

        """
        if not isinstance(value, type):
            raise InvalidParameterError("Parameter {} has invalid type, expecting {}.".format(key, type))

        return True
