"""Helper function to test if your server functions implementation runs correctly."""

import inspect
import json
from typing import Dict, List

import numpy as np

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.network.combiner.hooks.hooks import FunctionServiceServicer
from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase
from fedn.network.combiner.modelservice import bytesIO_request_generator, model_as_bytesIO


def test_server_functions(server_functions: ServerFunctionsBase, parameters_np: List[np.ndarray], client_metadata: Dict, rounds=5, num_clients=20):
    """Test if your functionalities are working on your server functions implementation.
    :param server_functions: An implementation of ServerFunctionsBase.
    :type server_functions: ServerFunctionsBase
    :param parameters: Model parameters in a list of numpy arrays.
    :type parameters: List[np.ndarray]
    """
    function_service = FunctionServiceServicer()
    function_code = inspect.getsource(server_functions)
    for i in range(rounds):
        print(f"Simulating server round: {i+1}")
        # see output from provided functions call
        request = fedn.ProvidedFunctionsRequest(function_code=function_code)
        _ = function_service.HandleProvidedFunctions(request, "")
        # see output from client selection request
        fake_clients = [str(j) for j in range(num_clients)]
        request = fedn.ClientSelectionRequest(client_ids=json.dumps(fake_clients))
        response = function_service.HandleClientSelection(request, "")
        selected_clients = json.loads(response.client_ids)
        # see output from client config request
        bytesio_model = model_as_bytesIO(parameters_np)
        request_function = fedn.ClientConfigRequest
        args = {}
        gen = bytesIO_request_generator(mdl=bytesio_model, request_function=request_function, args=args)
        function_service.HandleClientConfig(gen, "")
        # see output from aggregate request
        request_function = fedn.StoreModelRequest
        args = {"id": "global_model"}
        bytesio_model = model_as_bytesIO(parameters_np)
        gen = bytesIO_request_generator(mdl=bytesio_model, request_function=request_function, args=args)
        function_service.HandleStoreModel(gen, "")
        for k in range(len(selected_clients)):
            # send metadata
            client_id = selected_clients[k]
            request = fedn.ClientMetaRequest(metadata=json.dumps(client_metadata), client_id=client_id)
            function_service.HandleMetadata(request, "")
            request_function = fedn.StoreModelRequest
            args = {"id": client_id}
            bytesio_model = model_as_bytesIO(parameters_np)
            gen = bytesIO_request_generator(mdl=bytesio_model, request_function=request_function, args=args)
            function_service.HandleStoreModel(gen, "")
        request = fedn.AggregationRequest(aggregate="aggregate")
        response_generator = function_service.HandleAggregation(request, "")
        for response in response_generator:
            pass
        print(f"Server round {i+1} completed")
