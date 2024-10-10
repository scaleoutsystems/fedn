import io
import json
import os
import threading
import time
import uuid
from typing import Tuple

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.config import FEDN_CUSTOM_URL_PREFIX
from fedn.common.log_config import logger
from fedn.network.clients.client_api import ClientAPI, ConnectToApiResult, GrpcConnectionOptions
from fedn.network.combiner.modelservice import get_tmp_path
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
    def __init__(self,
            api_url: str,
            api_port: int,
            client_obj: ClientOptions,
            combiner_host: str = None,
            combiner_port: int = None,
            token: str = None,
            package_checksum: str = None,
            helper_type: str = None
        ):
        self.api_url = api_url
        self.api_port = api_port
        self.combiner_host = combiner_host
        self.combiner_port = combiner_port
        self.token = token
        self.client_obj = client_obj
        self.package_checksum = package_checksum
        self.helper_type = helper_type

        self.fedn_api_url = get_url(self.api_url, self.api_port)

        self.client_api: ClientAPI = ClientAPI()

        self.helper = None

    def _connect_to_api(self) -> Tuple[bool, dict]:
        result = None

        while not result or result == ConnectToApiResult.ComputePackgeMissing:
            if result == ConnectToApiResult.ComputePackgeMissing:
                logger.info("Retrying in 3 seconds")
                time.sleep(3)
            result, response = self.client_api.connect_to_api(self.fedn_api_url, self.token, self.client_obj.to_json())

        if result == ConnectToApiResult.Assigned:
            return True, response

        return False, None

    def start(self):
        if self.combiner_host and self.combiner_port:
            combiner_config = {
                "host": self.combiner_host,
                "port": self.combiner_port,
            }
        else:
            result, combiner_config = self._connect_to_api()
            if not result:
                return

        if self.client_obj.package == "remote":
            result = self.client_api.init_remote_compute_package(url=self.fedn_api_url, token=self.token, package_checksum=self.package_checksum)

            if not result:
                return
        else:
            result = self.client_api.init_local_compute_package()

            if not result:
                return

        self.set_helper(combiner_config)

        result: bool = self.client_api.init_grpchandler(config=combiner_config, client_name=self.client_obj.client_id, token=self.token)

        if not result:
            return

        logger.info("-----------------------------")

        threading.Thread(
            target=self.client_api.send_heartbeats, kwargs={"client_name": self.client_obj.name, "client_id": self.client_obj.client_id}, daemon=True
        ).start()

        self.client_api.subscribe("train", self.on_train)
        self.client_api.subscribe("validation", self.on_validation)

        threading.Thread(
            target=self.client_api.listen_to_task_stream, kwargs={"client_name": self.client_obj.name, "client_id": self.client_obj.client_id}, daemon=True
        ).start()

        stop_event = threading.Event()
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Client stopped by user.")

    def set_helper(self, response: GrpcConnectionOptions = None):
        helper_type = response.get("helper_type", None)

        helper_type_to_use = self.helper_type or helper_type or "numpyhelper"

        logger.info(f"Setting helper to: {helper_type_to_use}")

        # Priority: helper_type from constructor > helper_type from response > default helper_type
        self.helper = get_helper(helper_type_to_use)

    def on_train(self, request):
        logger.info("Received train request")
        self._process_training_request(request)

    def on_validation(self, request):
        logger.info("Received validation request")
        self._process_validation_request(request)


    def _process_training_request(self, request) -> Tuple[str, dict]:
        """Process a training (model update) request.

        :param model_id: The model id of the model to be updated.
        :type model_id: str
        :param session_id: The id of the current session
        :type session_id: str
        :return: The model id of the updated model, or None if the update failed. And a dict with metadata.
        :rtype: tuple
        """
        model_id: str = request.model_id
        session_id: str = request.session_id

        self.client_api.send_status(
            f"\t Starting processing of training request for model_id {model_id}",
            sesssion_id=session_id,
            sender_name=self.client_obj.name
        )

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

            logger.info("SETTING Training metadata: {}".format(training_metadata))
            meta["training_metadata"] = training_metadata

            os.unlink(inpath)
            os.unlink(outpath)
            os.unlink(outpath + "-metadata")

        except Exception as e:
            logger.error("Could not process training request due to error: {}".format(e))
            updated_model_id = None
            meta = {"status": "failed", "error": str(e)}

        if meta is not None:
            processing_time = time.time() - tic
            meta["processing_time"] = processing_time
            meta["config"] = request.data

        if model_id is not None:
            # Send model update to combiner

            self.client_api.send_model_update(
                sender_name=self.client_obj.name,
                sender_role=fedn.WORKER,
                client_id=self.client_obj.client_id,
                model_id=model_id,
                model_update_id=str(updated_model_id),
                receiver_name=request.sender.name,
                receiver_role=request.sender.role,
                meta=meta,
            )

            self.client_api.send_status(
                "Model update completed.",
                log_level=fedn.Status.AUDIT,
                type=fedn.StatusType.MODEL_UPDATE,
                request=request,
                sesssion_id=session_id,
                sender_name=self.client_obj.name
            )

    def _process_validation_request(self, request):
        """Process a validation request.

        :param model_id: The model id of the model to be validated.
        :type model_id: str
        :param session_id: The id of the current session.
        :type session_id: str
        :return: The validation metrics, or None if validation failed.
        :rtype: dict
        """
        model_id: str = request.model_id
        session_id: str = request.session_id
        cmd = "validate"

        self.client_api.send_status(f"Processing {cmd} request for model_id {model_id}", sesssion_id=session_id, sender_name=self.client_obj.name)

        try:
            model = self.client_api.get_model_from_combiner(id=str(model_id), client_name=self.client_obj.client_id)
            if model is None:
                logger.error("Could not retrieve model from combiner. Aborting validation request.")
                return
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(model.getbuffer())

            outpath = get_tmp_path()
            self.client_api.dispatcher.run_cmd(f"{cmd} {inpath} {outpath}")

            with open(outpath, "r") as fh:
                metrics = json.loads(fh.read())

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            logger.warning("Validation failed with exception {}".format(e))

        if metrics is not None:
            # Send validation
            validation = fedn.ModelValidation()
            validation.sender.name = self.client_obj.name
            validation.sender.role = fedn.WORKER
            validation.receiver.name = request.sender.name
            validation.receiver.role = request.sender.role
            validation.model_id = str(request.model_id)
            validation.data = json.dumps(metrics)
            validation.timestamp.GetCurrentTime()
            validation.correlation_id = request.correlation_id
            validation.session_id = request.session_id

            # sender_name: str, sender_role: fedn.Role, model_id: str, model_update_id: str
            result: bool = self.client_api.send_model_validation(
                sender_name=self.client_obj.name,
                receiver_name=request.sender.name,
                receiver_role=request.sender.role,
                model_id=str(request.model_id),
                metrics=json.dumps(metrics),
                correlation_id=request.correlation_id,
                session_id=request.session_id,
            )

            if result:
                self.client_api.send_status(
                    "Model validation completed.",
                    log_level=fedn.Status.AUDIT,
                    type=fedn.StatusType.MODEL_VALIDATION,
                    request=validation,
                    sesssion_id=request.session_id,
                    sender_name=self.client_obj.name
                )
            else:
                self.client_api.send_status(
                    "Client {} failed to complete model validation.".format(self.name),
                    log_level=fedn.Status.WARNING,
                    request=request,
                    sesssion_id=request.session_id,
                    sender_name=self.client_obj.name
                )
