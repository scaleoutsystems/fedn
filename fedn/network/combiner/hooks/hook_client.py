import json
import os
from io import BytesIO

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import bytesIO_request_generator, load_model_from_bytes, model_as_bytesIO
from fedn.network.combiner.updatehandler import UpdateHandler

CHUNK_SIZE = 1024 * 1024


class CombinerHookInterface:
    def __init__(self):
        logger.info("Starting hook client")
        self.hook_service_host = os.getenv("HOOK_SERVICE_HOST", "hook:12081")
        self.channel = grpc.insecure_channel(self.hook_service_host)
        self.stub = rpc.FunctionServiceStub(self.channel)

    def provided_functions(self, server_functions: str):
        """Communicates to hook container and asks which functions are available."""
        request = fedn.ProvidedFunctionsRequest(function_code=server_functions)

        response = self.stub.HandleProvidedFunctions(request)
        return response.available_functions

    def client_config(self, global_model) -> dict:
        """Communicates to hook container to get a client config."""
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
        """Aggregation call to the hook functions.

        Sends models in chunks, then asks for aggregation.
        """
        data = {}
        data["time_model_load"] = 0.0
        data["time_model_aggregation"] = 0.0
        # send previous global
        request_function = fedn.AggregationRequest
        args = {"id": "global_model", "aggregate": False}
        model = model_as_bytesIO(previous_global)
        self.stub.HandleAggregation(bytesIO_request_generator(mdl=model, request_function=request_function, args=args))
        # send client models and metadata
        updates = update_handler.get_model_updates()
        for update in updates:
            model, metadata = update_handler.load_model_update(update, helper)
            # send metadata
            client_id = update.sender.client_id
            request = fedn.ClientMetaRequest(metadata=metadata, client_id=client_id)
            self.stub.HandleMetadata(request)
            # send client model
            model = model_as_bytesIO(model)
            args = {"id": client_id, "aggregate": False}
            request_function = fedn.AggregationRequest
            self.stub.HandleAggregation(bytesIO_request_generator(mdl=model, request_function=request_function, args=args))
            if delete_models:
                # delete model from disk
                update_handler.delete_model(model_update=update)
        # ask for aggregation
        request = fedn.AggregationRequest(data=None, client_id="", aggregate=True)
        response = self.stub.HandleAggregation(request)
        data["nr_aggregated_models"] = len(updates)
        return load_model_from_bytes(response.data, helper), data
