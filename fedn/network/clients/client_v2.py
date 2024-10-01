import io
import json
import os
import threading
import time
import uuid
from typing import Tuple

from fedn.common.config import FEDN_CUSTOM_URL_PREFIX
from fedn.common.log_config import logger
from fedn.network.clients.client_api import ClientAPI, ConnectToApiResult, GrpcConnectionOptions
from fedn.utils.helpers.helpers import get_helper


def get_url(api_url: str, api_port: int) -> str:
    return f"{api_url}:{api_port}/{FEDN_CUSTOM_URL_PREFIX}" if api_port else f"{api_url}/{FEDN_CUSTOM_URL_PREFIX}"


class ClientOptions:
    def __init__(self, name: str, package: str, preferred_combiner: str = None, id: str = None):
        # check if name is a string and set. if not raise an error
        self._validate(name, package)
        self.name = name
        self.package = package
        self.preferred_combiner = preferred_combiner
        self.client_id = id if id else str(uuid.uuid4())

    def _validate(self, name: str, package: str):
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("Name must be a string")
        if not isinstance(package, str) or len(package) == 0 or package not in ["local", "remote"]:
            raise ValueError("Package must be either 'local' or 'remote'")

    # to json object
    def to_json(self):
        return {
            "name": self.name,
            "client_id": self.client_id,
            "preferred_combiner": self.preferred_combiner,
            "package": self.package,
        }


class Client:
    def __init__(self, api_url: str, api_port: int, client_obj: ClientOptions, token: str = None, package_checksum: str = None, helper_type: str = None):
        self.api_url = api_url
        self.api_port = api_port
        self.token = token
        self.client_obj = client_obj
        self.package_checksum = package_checksum
        self.helper_type = helper_type

        self.connect_string = get_url(self.api_url, self.api_port)
        logger.info(self.connect_string)

        self.client_api: ClientAPI = ClientAPI()

        self.helper = None

    def _connect_to_api(self) -> Tuple[bool, dict]:
        result, response = self.client_api.connect_to_api(self.connect_string, self.token, self.client_obj.to_json())

        if result == ConnectToApiResult.Assigned:
            return True, response
        elif result == ConnectToApiResult.ComputePackgeMissing:
            logger.info("Compute package not uploaded. Retrying in 3 seconds")
            time.sleep(3)
            return self._connect_to_api()

        logger.error(f"Error: {response}")
        return False, None

    def start(self):
        result, response = self._connect_to_api()

        if not result:
            return

        logger.info("Client assinged to controller")

        if self.client_obj.package == "remote":
            result = self.init_remote_compute_packae()

            if not result:
                return
        else:
            result = self.init_local_compute_package()

            if not result:
                return

        self.set_helper(response)

        result: bool = self.client_api.init_grpchandler(config=response, client_name=self.client_obj.client_id, token=self.token)

        if not result:
            logger.error("Could not initialize grpc handler")
            return

        logger.info("Client connected to grpc handler")

        logger.info("-----------------------------")

        # TODO: Check if thread is dead
        threading.Thread(
            target=self.client_api.send_heartbeats, kwargs={"client_name": self.client_obj.name, "client_id": self.client_obj.client_id}, daemon=True
        ).start()

        self.client_api.subscribe("train", self.on_train)

        threading.Thread(
            target=self.client_api.listen_to_task_stream, kwargs={"client_name": self.client_obj.name, "client_id": self.client_obj.client_id}, daemon=True
        ).start()

        while True:
            time.sleep(10)

    def set_helper(self, response: GrpcConnectionOptions = None):
        helper_type = response.get("helper_type", None)

        helper_type_to_use = self.helper_type or helper_type or "numpyhelper"

        logger.info(f"Setting helper to: {helper_type_to_use}")

        # Priority: helper_type from constructor > helper_type from response > default helper_type
        self.helper = get_helper(helper_type_to_use)

    def on_train(self, request):
        logger.info(f"Received train request: {request}")

        model_id: str = request.model_id
        session_id: str = request.session_id
        self._process_training_request(model_id, session_id)

    def init_remote_compute_packae(self) -> bool:
        result: bool = self.client_api.download_compute_package(self.connect_string, self.token)
        if not result:
            logger.error("Could not download compute package")
            return False
        result: bool = self.client_api.set_compute_package_checksum(self.connect_string, self.token)
        if not result:
            logger.error("Could not set checksum")
            return False

        if self.package_checksum:
            result: bool = self.client_api.validate_compute_package(self.package_checksum)
            if not result:
                logger.error("Could not validate compute package")
                return False

        result, path = self.client_api.unpack_compute_package()

        if not result:
            logger.error("Could not unpack compute package")
            return False

        logger.info(f"Compute package unpacked to: {path}")

        result = self.client_api.set_dispatcher(path)

        if not result:
            logger.error("Could not set dispatcher")
            return False

        logger.info("Dispatcher set")

        return True

    def init_local_compute_package(self):
        path = os.path.join(os.getcwd(), "client")
        result = self.client_api.set_dispatcher(path)

        if not result:
            logger.error("Could not set dispatcher")
            return False

        logger.info("Dispatcher set")

        return True

    def _process_training_request(self, model_id: str, session_id: str = None) -> Tuple[str, dict]:
        """Process a training (model update) request.

        :param model_id: The model id of the model to be updated.
        :type model_id: str
        :param session_id: The id of the current session
        :type session_id: str
        :return: The model id of the updated model, or None if the update failed. And a dict with metadata.
        :rtype: tuple
        """
        # self.send_status("\t Starting processing of training request for model_id {}".format(model_id), sesssion_id=session_id)

        try:
            meta = {}
            tic = time.time()

            model = self.client_api.get_model_from_combiner(id=str(model_id), client_name=self.client_obj.client_id)

            if model is None:
                logger.error("Could not retrieve model from combiner. Aborting training request.")
                return None, None

            meta["fetch_model"] = time.time() - tic

            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(model.getbuffer())

            outpath = self.helper.get_tmp_path()

            tic = time.time()
            # TODO: Check return status, fail gracefully

            self.client_api.dispatcher.run_cmd("train {} {}".format(inpath, outpath))

            meta["exec_training"] = time.time() - tic

            tic = time.time()
            out_model = None

            with open(outpath, "rb") as fr:
                out_model = io.BytesIO(fr.read())

            # Stream model update to combiner server
            updated_model_id = uuid.uuid4()
            self.client_api.send_model_to_combiner(out_model, str(updated_model_id))
            meta["upload_model"] = time.time() - tic

            # Read the metadata file
            with open(outpath + "-metadata", "r") as fh:
                training_metadata = json.loads(fh.read())
            meta["training_metadata"] = training_metadata

            os.unlink(inpath)
            os.unlink(outpath)
            os.unlink(outpath + "-metadata")

        except Exception as e:
            logger.error("Could not process training request due to error: {}".format(e))
            updated_model_id = None
            meta = {"status": "failed", "error": str(e)}

        return updated_model_id, meta
