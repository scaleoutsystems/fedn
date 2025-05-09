import ast
import json
from concurrent import futures

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger

# imports for user defined code
from fedn.network.combiner.hooks.allowed_import import *  # noqa: F403
from fedn.network.combiner.hooks.allowed_import import ServerFunctionsBase
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
        self.implemented_functions = {}
        logger.info("Server Functions initialized.")

    def HandleClientConfig(self, request_iterator: fedn.ClientConfigRequest, context):
        """Distribute client configs to clients from user defined code.

        :param request_iterator: the client config request
        :type request_iterator: :class:`fedn.network.grpc.fedn_pb2.ClientConfigRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client config response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ClientConfigResponse`
        """
        try:
            logger.info("Received client config request.")
            model, _ = unpack_model(request_iterator, self.helper)
            client_settings = self.server_functions.client_settings(global_model=model)
            logger.info(f"Client config response: {client_settings}")
            return fedn.ClientConfigResponse(client_settings=json.dumps(client_settings))
        except Exception as e:
            logger.error(f"Error handling client config request: {e}")

    def HandleClientSelection(self, request: fedn.ClientSelectionRequest, context):
        """Handle client selection from user defined code.

        :param request: the client selection request
        :type request: :class:`fedn.network.grpc.fedn_pb2.fedn.ClientSelectionRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client selection response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ClientSelectionResponse`
        """
        try:
            logger.info("Received client selection request.")
            client_ids = json.loads(request.client_ids)
            client_ids = self.server_functions.client_selection(client_ids)
            logger.info(f"Clients selected: {client_ids}")
            return fedn.ClientSelectionResponse(client_ids=json.dumps(client_ids))
        except Exception as e:
            logger.error(f"Error handling client selection request: {e}")

    def HandleMetadata(self, request: fedn.ClientMetaRequest, context):
        """Store client metadata from a request.

        :param request: the client meta request
        :type request: :class:`fedn.network.grpc.fedn_pb2.fedn.ClientMetaRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client meta response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ClientMetaResponse`
        """
        try:
            logger.info("Received metadata")
            client_id = request.client_id
            metadata = json.loads(request.metadata)
            # dictionary contains: [model, client_metadata] in that order for each key
            self.client_updates[client_id] = self.client_updates.get(client_id, []) + [metadata]
            self.check_incremental_aggregate(client_id)
            return fedn.ClientMetaResponse(status="Metadata stored")
        except Exception as e:
            logger.error(f"Error handling store metadata request: {e}")

    def HandleStoreModel(self, request_iterator, context):
        try:
            model, final_request = unpack_model(request_iterator, self.helper)
            client_id = final_request.id
            if client_id == "global_model":
                logger.info("Received previous global model")
                self.previous_global = model
            else:
                logger.info(f"Received client model from client {client_id}")
                # dictionary contains: [model, client_metadata] in that order for each key
                self.client_updates[client_id] = [model] + self.client_updates.get(client_id, [])
            self.check_incremental_aggregate(client_id)
            return fedn.StoreModelResponse(status=f"Received model originating from {client_id}")
        except Exception as e:
            logger.error(f"Error handling store model request: {e}")

    def check_incremental_aggregate(self, client_id):
        # incremental aggregation (memory secure)
        if client_id == "global_model":
            return
        model_and_metadata_received = len(self.client_updates[client_id]) == 2
        if model_and_metadata_received and self.implemented_functions["incremental_aggregate"]:
            client_model = self.client_updates[client_id][0]
            client_metadata = self.client_updates[client_id][1]
            self.server_functions.incremental_aggregate(client_id, client_model, client_metadata, self.previous_global)
            del self.client_updates[client_id]

    def HandleAggregation(self, request, context):
        """Receive and store models and aggregate based on user-defined code when specified in the request.

        :param request_iterator: the aggregation request
        :type request_iterator: :class:`fedn.network.grpc.fedn_pb2.fedn.AggregationRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the aggregation response (aggregated model or None)
        :rtype: :class:`fedn.network.grpc.fedn_pb2.AggregationResponse`
        """
        try:
            logger.info(f"Receieved aggregation request: {request.aggregate}")
            if self.implemented_functions["incremental_aggregate"]:
                aggregated_model = self.server_functions.get_incremental_aggregate_model()
            else:
                aggregated_model = self.server_functions.aggregate(self.previous_global, self.client_updates)

            model_bytesIO = model_as_bytesIO(aggregated_model, self.helper)
            request_function = fedn.AggregationResponse
            self.client_updates = {}
            logger.info("Returning aggregate model.")
            response_generator = bytesIO_request_generator(mdl=model_bytesIO, request_function=request_function, args={})
            for response in response_generator:
                yield response
        except Exception as e:
            logger.error(f"Error handling aggregation request: {e}")

    def HandleProvidedFunctions(self, request: fedn.ProvidedFunctionsResponse, context):
        """Handles the 'provided_functions' request. Sends back which functions are available.

        :param request: the provided function request
        :type request: :class:`fedn.network.grpc.fedn_pb2.fedn.ProvidedFunctionsRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: dict with str -> bool for which functions are available
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ProvidedFunctionsResponse`
        """
        try:
            logger.info("Receieved provided functions request.")
            server_functions_code = request.function_code
            # if no new code return previous
            if server_functions_code == self.server_functions_code:
                logger.info("No new server function code provided.")
                logger.info(f"Provided functions: {self.implemented_functions}")
                return fedn.ProvidedFunctionsResponse(available_functions=self.implemented_functions)

            self.server_functions_code = server_functions_code
            self.implemented_functions = {}
            self._instansiate_server_functions_code()
            functions = ["client_selection", "client_settings", "aggregate", "incremental_aggregate"]
            # parse the entire code string into an AST
            tree = ast.parse(server_functions_code)

            # collect all real function names
            defined_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

            # check each target function
            for func in functions:
                if func in defined_funcs:
                    print(f"Function '{func}' found—assuming it´s implemented.")
                    self.implemented_functions[func] = True
                else:
                    print(f"Function '{func}' not found.")
                    self.implemented_functions[func] = False
            logger.info(f"Provided function: {self.implemented_functions}")
            return fedn.ProvidedFunctionsResponse(available_functions=self.implemented_functions)
        except Exception as e:
            logger.error(f"Error handling provided functions request: {e}")

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
    KEEPALIVE_TIME_MS = 5 * 60 * 1000  # send keepalive ping every 5 minutes
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
