import json
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

    def call_function_service(self, task, payload, meta={}):
        if task == "setup" or task == "get_model_metadata":
            request = fedn.FunctionRequest(task=task, payload_string=payload)
        if task == "aggregate" or task == "store_parameters":
            request = fedn.FunctionRequest(task=task, payload_bytes=payload, payload_string=json.dumps(meta))
        try:
            response = self.stub.ExecuteFunction(request)
            return response
        except grpc.RpcError as e:
            logger.info(f"RPC failed: {e}")
            return None

    def set_function_provider(self, class_code):
        if not isinstance(class_code, str):
            raise TypeError("class_code must be of type string")
        self.call_function_service("setup", class_code)
        logger.info("Function provider code set.")

    def call_function(self, task, payload, helper, meta):
        if task == "aggregate":
            payload = serialize_model_to_BytesIO(payload, helper).getvalue()
            response = self.call_function_service(task, payload, meta=meta)
            return load_model_from_BytesIO(response.result_bytes, helper)
        if task == "store_parameters":
            payload = serialize_model_to_BytesIO(payload, helper).getvalue()
            response = self.call_function_service(task, payload, meta=meta)

    def get_model_metadata(self):
        response = self.call_function_service("get_model_metadata", "")
        model_metadata_json = response.result_string
        return json.loads(model_metadata_json)
