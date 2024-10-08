import json
from concurrent import futures
from io import BytesIO

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger

# imports for user code
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random  # noqa: F401
from fedn.network.combiner.modelservice import bytesIO_request_generator, load_model_from_bytes, model_as_bytesIO
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
        model = self.unpack_model(request_iterator)
        client_config = self.server_functions.client_config(global_model=model)
        logger.info(f"Client config response: {client_config}")
        return fedn.ClientConfigResponse(client_config=json.dumps(client_config))

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
        client_id = request.client_id
        metadata = json.loads(request.metadata)
        self.client_updates[client_id] = self.client_updates.get(client_id, []) + [metadata]
        return fedn.ClientMetaResponse(status="Metadata stored")

    def HandleAggregation(self, request_iterator: fedn.AggregationRequest, context):
        """Receive and store models and aggregate based on user-defined code when specified in the request.

        :param request_iterator: the aggregation request
        :type request_iterator: :class:`fedn.network.grpc.fedn_pb2.fedn.AggregationRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the aggregation response (aggregated model or None)
        :rtype: :class:`fedn.network.grpc.fedn_pb2.AggregationResponse`
        """
        # check what type of request
        for request in request_iterator:
            if request.aggregate:
                logger.info("Received aggregation request.")
                aggregated_model = self.server_functions.aggregate(self.previous_global, self.client_updates)
                aggregated_model = model_as_bytesIO(aggregated_model)
                request_function = fedn.AggregationResponse
                logger.info("Returning aggregate model.")
                return bytesIO_request_generator(mdl=aggregated_model, request_function=request_function, args={})
            client_id = request.client_id
            break
        logger.info(f"Received request to store model originating from: {client_id}")
        model = self.unpack_model(request_iterator)

        if client_id == "global_model":
            self.previous_global = model
        else:
            self.client_updates[client_id] = [model] + self.client_updates.get(client_id, [])
        return fedn.AggregationResponse(data=None)

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
        server_functions_code = request.function_code
        if self.server_functions is None or server_functions_code != self.server_functions_code:
            self.server_functions_code = server_functions_code
            # this will create a new user defined instance of the ServerFunctions class.
            try:
                namespace = {}
                exec(server_functions_code, globals(), namespace)  # noqa: S102
                exec("server_functions = ServerFunctions()", globals(), namespace)  # noqa: S102
                self.server_functions = namespace.get("server_functions")
            except Exception as e:
                logger.error(f"Exec failed with error: {str(e)}")
        implemented_functions = {}
        # if crashed or not returning None we assume function is implemented
        # check if aggregation is available
        try:
            ret = self.server_functions.aggregate(0, 0)
            if ret is None:
                implemented_functions["aggregate"] = False
            else:
                implemented_functions["aggregate"] = True
        except Exception:
            implemented_functions["aggregate"] = True
        # check if client_config is available
        try:
            ret = self.server_functions.client_config(0)
            if ret is None:
                implemented_functions["client_config"] = False
            else:
                implemented_functions["client_config"] = True
        except Exception:
            implemented_functions["client_config"] = True
        # check if client_selection is available
        try:
            ret = self.server_functions.client_selection(0)
            if ret is None:
                implemented_functions["client_selection"] = False
            else:
                implemented_functions["client_selection"] = True
        except Exception:
            implemented_functions["client_selection"] = True
        logger.info(f"Provided function: {implemented_functions}")
        return fedn.ProvidedFunctionsResponse(available_functions=implemented_functions)

    def unpack_model(self, request_iterator):
        """Unpack the incoming model sent in chunks from the request iterator.

        :param request_iterator: A streaming iterator from the gRPC service.
        :return: The reconstructed model parameters.
        """
        model_buffer = BytesIO()
        try:
            for request in request_iterator:
                if request.data:
                    model_buffer.write(request.data)
        except MemoryError as e:
            print(f"Memory error occured when loading model, reach out to the FEDn team if you need a solution to this. {e}")
            raise
        except Exception as e:
            print(f"Exception occured during model loading: {e}")
            raise

        model_buffer.seek(0)

        model_bytes = model_buffer.getvalue()

        return load_model_from_bytes(model_bytes, self.helper)


def serve():
    """Start the hooks service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_FunctionServiceServicer_to_server(FunctionServiceServicer(), server)
    server.add_insecure_port("[::]:12081")
    server.start()
    server.wait_for_termination()
