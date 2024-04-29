import os
import unittest

from fedn.common.exceptions import InvalidParameterError
from fedn.utils.parameters import Parameters


class TestHelpers(unittest.TestCase):

    # Test an invalid paramter name

    def test_parameters_invalidkey(self):

        parameters = {
            'serverop': 'adam',
            'learning_rate': 1e-3,
        }

        parameter_schema = {
            'serveropt': str,
            'learning_rate': float,
        }

        self.assertRaises(InvalidParameterError, Parameters.validate_parameters, parameters, parameter_schema)

    def test_parameters_valid(self):

        parameters = {
            'serveropt': 'adam',
            'learning_rate': 1e-3,
            'beta1': 0.9,
            'beta2': 0.99,
            'tau': 1e-4,
        }

        parameter_schema = {
            'serveropt': str,
            'learning_rate': float,
            'beta1': float,
            'beta2': float,
            'tau': float,
        }

        self.assertTrue(Parameters.validate_parameters(parameters, parameter_schema))

    def test_parameters_invalid(self):

        parameters = {
            'serveropt': 'adam',
            'learning_rate': 1e-3,
            'beta1': 0.9,
            'beta2': 0.99,
            'tau': 1e-4,
        }

        parameter_schema = {
            'serveropt': str,
            'learning_rate': float,
            'beta1': float,
            'beta2': str,
            'tau': float,
        }

        self.assertRaises(InvalidParameterError, Parameters.validate_parameters, parameters, parameter_schema)


if __name__ == '__main__':
    unittest.main()
