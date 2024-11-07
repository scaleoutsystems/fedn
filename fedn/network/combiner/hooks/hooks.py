import json
from concurrent import futures

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger

# imports for user code
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random  # noqa: F401
from fedn.network.combiner.modelservice import bytesIO_request_generator, model_as_bytesIO, unpack_model
from fedn.utils.helpers.plugins.numpyhelper import Helper

CHUNK_SIZE = 1024 * 1024
VALID_NAME_REGEX = "^[a-zA-Z0-9_-]*$"


class FunctionServiceServicer(rpc.FunctionServiceServicer):
    """Function service running in an environment combined with each combiner.

    Receiving requests from the combiner.
    """

    def __init__(self) -> None:
        """Initialize long-running Function server."""
        super().__init__()

        self.helper = Helper()
        self.server_functions: ServerFunctionsBase = None
        self.server_functions_code: str = None
        self.client_updates = {}
        self.implemented_functions = None

    def HandleClientConfig(self, request_iterator: fedn.ClientConfigRequest, context):
        """Distribute client configs to clients from user defined code.

        :param request_iterator: the client config request
        :type request_iterator: :class:`fedn.network.grpc.fedn_pb2.ClientConfigRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client config response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ClientConfigResponse`
        """
        logger.info("Received client config request.")
        model, _ = unpack_model(request_iterator, self.helper)
        client_settings = self.server_functions.client_settings(global_model=model)
        logger.info(f"Client config response: {client_settings}")
        return fedn.ClientConfigResponse(client_settings=json.dumps(client_settings))

    def HandleClientSelection(self, request: fedn.ClientSelectionRequest, context):
        """Handle client selection from user defined code.

        :param request: the client selection request
        :type request: :class:`fedn.network.grpc.fedn_pb2.fedn.ClientSelectionRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client selection response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ClientSelectionResponse`
        """
        logger.info("Received client selection request.")
        client_ids = json.loads(request.client_ids)
        client_ids = self.server_functions.client_selection(client_ids)
        logger.info(f"Clients selected: {client_ids}")
        return fedn.ClientSelectionResponse(client_ids=json.dumps(client_ids))

    def HandleMetadata(self, request: fedn.ClientMetaRequest, context):
        """Store client metadata from a request.

        :param request: the client meta request
        :type request: :class:`fedn.network.grpc.fedn_pb2.fedn.ClientMetaRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client meta response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ClientMetaResponse`
        """
        logger.info("Received metadata")
        client_id = request.client_id
        metadata = json.loads(request.metadata)
        self.client_updates[client_id] = self.client_updates.get(client_id, []) + [metadata]
        return fedn.ClientMetaResponse(status="Metadata stored")

    def HandleStoreModel(self, request_iterator, context):
        model, final_request = unpack_model(request_iterator, self.helper)
        client_id = final_request.id
        if client_id == "global_model":
            logger.info("Received previous global model")
            self.previous_global = model
        else:
            logger.info("Received client model")
            self.client_updates[client_id] = [model] + self.client_updates.get(client_id, [])
        return fedn.StoreModelResponse(status=f"Received model originating from {client_id}")

    def HandleAggregation(self, request, context):
        """Receive and store models and aggregate based on user-defined code when specified in the request.

        :param request_iterator: the aggregation request
        :type request_iterator: :class:`fedn.network.grpc.fedn_pb2.fedn.AggregationRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the aggregation response (aggregated model or None)
        :rtype: :class:`fedn.network.grpc.fedn_pb2.AggregationResponse`
        """
        logger.info(f"Receieved aggregation request: {request.aggregate}")
        aggregated_model = self.server_functions.aggregate(self.previous_global, self.client_updates)
        model_bytesIO = model_as_bytesIO(aggregated_model, self.helper)
        request_function = fedn.AggregationResponse
        self.client_updates = {}
        logger.info("Returning aggregate model.")
        response_generator = bytesIO_request_generator(mdl=model_bytesIO, request_function=request_function, args={})
        for response in response_generator:
            yield response

    def HandleProvidedFunctions(self, request: fedn.ProvidedFunctionsResponse, context):
        """Handles the 'provided_functions' request. Sends back which functions are available.

        :param request: the provided function request
        :type request: :class:`fedn.network.grpc.fedn_pb2.fedn.ProvidedFunctionsRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: dict with str -> bool for which functions are available
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ProvidedFunctionsResponse`
        """
        logger.info("Receieved provided functions request.")
        if self.implemented_functions is not None:
            return fedn.ProvidedFunctionsResponse(available_functions=self.implemented_functions)
        server_functions_code = request.function_code
        self.server_functions_code = server_functions_code
        self.implemented_functions = {}
        self._instansiate_server_functions_code()
        # if crashed or not returning None we assume function is implemented
        # check if aggregation is available
        try:
            ret = self.server_functions.aggregate(0, 0)
            if ret is None:
                self.implemented_functions["aggregate"] = False
            else:
                self.implemented_functions["aggregate"] = True
        except Exception:
            self.implemented_functions["aggregate"] = True
        # check if client_settings is available
        try:
            ret = self.server_functions.client_settings(0)
            if ret is None:
                self.implemented_functions["client_settings"] = False
            else:
                self.implemented_functions["client_settings"] = True
        except Exception:
            self.implemented_functions["client_settings"] = True
        # check if client_selection is available
        try:
            ret = self.server_functions.client_selection(0)
            if ret is None:
                self.implemented_functions["client_selection"] = False
            else:
                self.implemented_functions["client_selection"] = True
        except Exception:
            self.implemented_functions["client_selection"] = True
        logger.info(f"Provided function: {self.implemented_functions}")
        return fedn.ProvidedFunctionsResponse(available_functions=self.implemented_functions)

    def _instansiate_server_functions_code(self):
        # this will create a new user defined instance of the ServerFunctions class.
        try:
            namespace = {}
            exec(self.server_functions_code, globals(), namespace)  # noqa: S102
            exec("server_functions = ServerFunctions()", globals(), namespace)  # noqa: S102
            self.server_functions = namespace.get("server_functions")
        except Exception as e:
            logger.error(f"Exec failed with error: {str(e)}")


def serve():
    """Start the hooks service."""
    # Keepalive settings: these detect if the client is alive
    KEEPALIVE_TIME_MS = 60 * 1000  # send keepalive ping every 60 seconds
    KEEPALIVE_TIMEOUT_MS = 20 * 1000  # wait 20 seconds for keepalive ping ack before considering connection dead
    MAX_CONNECTION_IDLE_MS = 5 * 60 * 1000  # max idle time before server terminates the connection (5 minutes)
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB in bytes
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=100),  # Increase based on expected load
        options=[
            ("grpc.keepalive_time_ms", KEEPALIVE_TIME_MS),
            ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
            ("grpc.max_connection_idle_ms", MAX_CONNECTION_IDLE_MS),
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    rpc.add_FunctionServiceServicer_to_server(FunctionServiceServicer(), server)
    server.add_insecure_port("[::]:12081")
    server.start()
    server.wait_for_termination()
