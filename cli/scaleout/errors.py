class InvalidConfigurationError(Exception):
	""" Exception raised for errors in the project.yaml definition file. """

class AuthenticationError(Exception):
	""" Exception raised when studioclinet fails to connect. """	