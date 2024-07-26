from concurrent import futures

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import load_model_from_BytesIO, serialize_model_to_BytesIO
from fedn.network.hooks.safe_builtins import safe_builtins
from fedn.utils.helpers.plugins.numpyhelper import Helper

CHUNK_SIZE = 1024 * 1024
VALID_NAME_REGEX = "^[a-zA-Z0-9_-]*$"


class FunctionServiceServicer(rpc.FunctionServiceServicer):
    def __init__(self) -> None:
        super().__init__()
        self.safe_builtins = safe_builtins

        self.globals_dict = {
            "__builtins__": self.safe_builtins,
        }
        self.helper = Helper()
        self.parameters = []

    def ExecuteFunction(self, request, context):
        # Compile the function code
        if request.task == "setup":
            logger.info("Adding function provider.")
            self.init_hook_object(request.payload_string)
            return fedn.FunctionResponse(result_string="Instansiated hook functions.")
        if request.task == "store_parameters":
            logger.info("Executing aggregate function.")
            payload = load_model_from_BytesIO(request.payload_bytes, self.helper)
            self.parameters.append(payload)
            return fedn.FunctionResponse(result_string="Stored parameters")
        if request.task == "aggregate":
            logger.info("Executing aggregate function.")
            payload = load_model_from_BytesIO(request.payload_bytes, self.helper)
            self.parameters.append(payload)
            result = self.execute_function_code()
            result_bytes = serialize_model_to_BytesIO(result, self.helper).getvalue()
            return fedn.FunctionResponse(result_bytes=result_bytes)

    def init_hook_object(self, class_code):
        # Prepare the globals dictionary with restricted builtins

        # Compile and execute the class code
        exec(class_code, self.globals_dict)  # noqa: S102
        # Instantiate the object within restricted scope
        instance_code = """
function_provider = FunctionProvider()
"""
        # Compile and execute the instance code
        exec(instance_code, self.globals_dict)  # noqa: S102

    def execute_function_code(self):
        if not hasattr(self, "globals_dict"):
            raise AttributeError("Function provider code has not been provided.")
        self.globals_dict["parameters"] = self.parameters
        instance_code = """
res = function_provider.aggregate(parameters)
"""
        exec(instance_code, self.globals_dict)  # noqa: S102
        self.parameters = []
        return self.globals_dict["res"]


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_FunctionServiceServicer_to_server(FunctionServiceServicer(), server)
    server.add_insecure_port("[::]:12081")
    server.start()
    server.wait_for_termination()
