import io
import json
import os
import time
import uuid
from io import BytesIO
from typing import Tuple

from fedn.common.config import FEDN_CUSTOM_URL_PREFIX
from fedn.common.log_config import logger
from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient, GrpcConnectionOptions
from fedn.network.combiner.modelservice import get_tmp_path
from fedn.utils.helpers.helpers import get_helper, save_metadata


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
    def __init__(
        self,
        api_url: str,
        api_port: int,
        client_obj: ClientOptions,
        combiner_host: str = None,
        combiner_port: int = None,
        token: str = None,
        package_checksum: str = None,
        helper_type: str = None,
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

        self.fedn_client: FednClient = FednClient()

        self.helper = None

    def _connect_to_api(self) -> Tuple[bool, dict]:
        result = None

        while not result or result == ConnectToApiResult.ComputePackageMissing:
            if result == ConnectToApiResult.ComputePackageMissing:
                logger.info("Retrying in 3 seconds")
                time.sleep(3)
            result, response = self.fedn_client.connect_to_api(self.fedn_api_url, self.token, self.client_obj.to_json())

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
            result = self.fedn_client.init_remote_compute_package(url=self.fedn_api_url, token=self.token, package_checksum=self.package_checksum)

            if not result:
                return
        else:
            result = self.fedn_client.init_local_compute_package()

            if not result:
                return

        self.set_helper(combiner_config)

        result: bool = self.fedn_client.init_grpchandler(config=combiner_config, client_name=self.client_obj.client_id, token=self.token)

        if not result:
            return

        logger.info("-----------------------------")

        self.fedn_client.set_train_callback(self.on_train)
        self.fedn_client.set_validate_callback(self.on_validation)

        self.fedn_client.set_name(self.client_obj.name)
        self.fedn_client.set_client_id(self.client_obj.client_id)

        self.fedn_client.run()

    def set_helper(self, response: GrpcConnectionOptions = None):
        helper_type = response.get("helper_type", None)

        helper_type_to_use = self.helper_type or helper_type or "numpyhelper"

        logger.info(f"Setting helper to: {helper_type_to_use}")

        # Priority: helper_type from constructor > helper_type from response > default helper_type
        self.helper = get_helper(helper_type_to_use)

    def on_train(self, in_model, client_settings):
        out_model, meta = self._process_training_request(in_model, client_settings)
        return out_model, meta

    def on_validation(self, in_model):
        metrics = self._process_validation_request(in_model)
        return metrics

    def _process_training_request(self, in_model: BytesIO, client_settings: dict) -> Tuple[BytesIO, dict]:
        """Process a training (model update) request.

        :param in_model: The model to be updated.
        :type in_model: BytesIO
        :return: The updated model, or None if the update failed. And a dict with metadata.
        :rtype: tuple
        """
        try:
            meta = {}

            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(in_model.getbuffer())

            save_metadata(metadata=client_settings, filename=inpath)

            outpath = self.helper.get_tmp_path()

            tic = time.time()

            self.fedn_client.dispatcher.run_cmd("train {} {}".format(inpath, outpath))

            meta["exec_training"] = time.time() - tic

            out_model = None

            with open(outpath, "rb") as fr:
                out_model = io.BytesIO(fr.read())

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
            out_model = None
            meta = {"status": "failed", "error": str(e)}

        return out_model, meta

    def _process_validation_request(self, in_model: BytesIO) -> dict:
        """Process a validation request.

        :param in_model: The model to be validated.
        :type in_model: BytesIO
        :return: The validation metrics, or None if validation failed.
        :rtype: dict
        """
        try:
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(in_model.getbuffer())

            outpath = get_tmp_path()
            self.fedn_client.dispatcher.run_cmd(f"validate {inpath} {outpath}")

            with open(outpath, "r") as fh:
                metrics = json.loads(fh.read())

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            logger.warning("Validation failed with exception {}".format(e))
            metrics = None

        return metrics
