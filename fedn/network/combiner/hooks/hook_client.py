import json
import os

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import bytesIO_request_generator, model_as_bytesIO, unpack_model
from fedn.network.combiner.updatehandler import UpdateHandler

CHUNK_SIZE = 1024 * 1024


class CombinerHookInterface:
    """Combiner to server function hooks client."""

    def __init__(self):
        """Initialize CombinerHookInterface client."""
        self.hook_service_host = os.getenv("HOOK_SERVICE_HOST", "hook:12081")
        self.channel = grpc.insecure_channel(self.hook_service_host)
        self.stub = rpc.FunctionServiceStub(self.channel)

    def provided_functions(self, server_functions: str):
        """Communicates to hook container and asks which functions are available.

        :param server_functions: String version of an implementation of the ServerFunctionsBase interface.
        :type server_functions: :str:
        :return: dictionary specifing which functions are implemented.
        :rtype: dict
        """
        request = fedn.ProvidedFunctionsRequest(function_code=server_functions)

        response = self.stub.HandleProvidedFunctions(request)
        return response.available_functions

    def client_config(self, global_model) -> dict:
        """Communicates to hook container to get a client config.

        :param global_model: The global model that will be distributed to clients.
        :type global_model: :bytes:
        :return: config that will be distributed to clients.
        :rtype: dict
        """
        request_function = fedn.ClientConfigRequest
        args = {}
        model = model_as_bytesIO(global_model)
        response = self.stub.HandleClientConfig(bytesIO_request_generator(mdl=model, request_function=request_function, args=args))
        return json.loads(response.client_config)

    def client_selection(self, clients: list) -> list:
        request = fedn.ClientSelectionRequest(client_ids=json.dumps(clients))
        response = self.stub.HandleClientSelection(request)
        return json.loads(response.client_ids)

    def aggregate(self, previous_global, update_handler: UpdateHandler, helper, delete_models: bool):
        """Aggregation call to the hook functions. Sends models in chunks, then asks for aggregation.

        :param global_model: The global model that will be distributed to clients.
        :type global_model: :bytes:
        :return: config that will be distributed to clients.
        :rtype: dict
        """
        data = {}
        data["time_model_load"] = 0.0
        data["time_model_aggregation"] = 0.0
        # send previous global
        request_function = fedn.StoreModelRequest
        args = {"id": "global_model"}
        response = self.stub.HandleStoreModel(bytesIO_request_generator(mdl=previous_global, request_function=request_function, args=args))
        logger.info(f"Store model response: {response.status}")
        # send client models and metadata
        updates = update_handler.get_model_updates()
        for update in updates:
            metadata = json.loads(update.meta)["training_metadata"]
            model = update_handler.load_model_update_bytesIO(update.model_update_id)
            # send metadata
            client_id = update.sender.client_id
            request = fedn.ClientMetaRequest(metadata=json.dumps(metadata), client_id=client_id)
            response = self.stub.HandleMetadata(request)
            # send client model
            args = {"id": client_id}
            request_function = fedn.StoreModelRequest
            response = self.stub.HandleStoreModel(bytesIO_request_generator(mdl=model, request_function=request_function, args=args))
            logger.info(f"Store model response: {response.status}")
            if delete_models:
                # delete model from disk
                update_handler.delete_model(model_update=update)
        # ask for aggregation
        request = fedn.AggregationRequest(aggregate="aggregate")
        response_generator = self.stub.HandleAggregation(request)
        data["nr_aggregated_models"] = len(updates)
        model, _ = unpack_model(response_generator, helper)
        return model, data
