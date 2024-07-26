import os

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import load_model_from_BytesIO, serialize_model_to_BytesIO


class CombinerHookClient:
    def __init__(self):
        logger.info("Starting hook client")
        self.hook_service_host = os.getenv("HOOK_SERVICE_HOST", "hook:12081")
        self.channel = grpc.insecure_channel(self.hook_service_host)
        self.stub = rpc.FunctionServiceStub(self.channel)

    def call_function_service(self, task, payload):
        request = fedn.FunctionRequest(task=task, payload_string=payload) if task == "setup" else fedn.FunctionRequest(task=task, payload_bytes=payload)
        try:
            response = self.stub.ExecuteFunction(request)
            return response
        except grpc.RpcError as e:
            logger.info(f"RPC failed: {e}")
            return None

    # Example method to trigger function execution
    def set_function_provider(self, class_code):
        if not isinstance(class_code, str):
            raise TypeError("class_code must be of type string")
        self.call_function_service("setup", class_code)

    def call_function(self, task, payload, helper):
        if task == "aggregate":
            payload = serialize_model_to_BytesIO(payload, helper).getvalue()
            response = self.call_function_service(task, payload)
            return load_model_from_BytesIO(response.result_bytes, helper)
        if task == "store_parameters":
            payload = serialize_model_to_BytesIO(payload, helper).getvalue()
            response = self.call_function_service(task, payload)
