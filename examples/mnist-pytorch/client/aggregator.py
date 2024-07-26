import numpy as np


class FunctionProvider:
    def __init__(self) -> None:
        pass

    def aggregate(self, parameters):
        if len(parameters) == 0:
            return []
        num_clients = len(parameters)

        summed_parameters = [np.zeros_like(param) for param in parameters[0]]

        for client_params in parameters:
            for i, param in enumerate(client_params):
                summed_parameters[i] += param

        averaged_parameters = [param / num_clients for param in summed_parameters]

        return averaged_parameters
