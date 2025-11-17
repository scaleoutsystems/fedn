"""Client module for handling client operations in the FEDn network."""

import io
import json
import os
import tempfile
import time
from io import BytesIO
from typing import Optional, Tuple

from scaleout.client.connect import ClientOptions, get_url
from scaleoututil.logging import FednLogger
from scaleout.client.dispatcher_package_runtime import DispatcherPackageRuntime
from scaleout.client.fedn_client import ConnectToApiResult, FednClient, GrpcConnectionOptions
from scaleout.client.package_runtime import get_compute_package_dir_path
from scaleoututil.helpers.helpers import get_helper, save_metadata


def get_tmp_path():
    """Return a temporary output path compatible with save_model, load_model."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    return path


class DispatcherClient:
    """Client for interacting with the FEDn network."""

    def __init__(
        self,
        api_url: str,
        client_obj: ClientOptions,
        combiner_host: Optional[str] = None,
        combiner_port: Optional[int] = None,
        token: Optional[str] = None,
        package_checksum: Optional[str] = None,
        helper_type: Optional[str] = None,
    ) -> None:
        """Initialize the Client."""
        self.api_url = api_url
        self.combiner_host = combiner_host
        self.combiner_port = combiner_port
        self.token = token
        self.client_obj = client_obj
        self.package_checksum = package_checksum
        self.helper_type = helper_type

        package_path, archive_path = get_compute_package_dir_path()
        self._package_runtime = DispatcherPackageRuntime(package_path, archive_path)

        self.fedn_api_url = get_url(self.api_url)
        self.fedn_client: FednClient = FednClient()
        self.helper = None

    def _connect_to_api(self) -> Tuple[bool, Optional[dict]]:
        """Connect to the API and handle retries."""
        result = None
        response = None

        while not result or result == ConnectToApiResult.ComputePackageMissing:
            if result == ConnectToApiResult.ComputePackageMissing:
                FednLogger().info("Retrying in 3 seconds")
                time.sleep(3)
            result, response = self.fedn_client.connect_to_api(self.fedn_api_url, self.token, self.client_obj.to_json())

        if result == ConnectToApiResult.Assigned:
            return True, response

        return False, None

    def start(self) -> None:
        """Start the client."""
        if self.combiner_host and self.combiner_port:
            combiner_config = GrpcConnectionOptions(host=self.combiner_host, port=self.combiner_port)
        else:
            result, combiner_config = self._connect_to_api()
            if not result:
                return
        if self.client_obj.package == "remote":
            result = self._package_runtime.load_remote_compute_package(url=self.fedn_api_url, token=self.token)
            if not result:
                return
        else:
            result = self._package_runtime.load_local_compute_package(os.path.join(os.getcwd(), "client"))
            if not result:
                return

        result = self._package_runtime.run_startup()
        if not result:
            return

        self.set_helper(combiner_config)

        result = self.fedn_client.init_grpchandler(config=combiner_config, client_id=self.client_obj.client_id, token=self.token)
        if not result:
            return

        FednLogger().info("-----------------------------")

        self.fedn_client.set_train_callback(self.on_train)
        self.fedn_client.set_validate_callback(self.on_validation)
        self.fedn_client.set_forward_callback(self.on_forward)
        self.fedn_client.set_backward_callback(self.on_backward)
        self.fedn_client.set_predict_callback(self._process_prediction_request)

        self.fedn_client.set_name(self.client_obj.name)
        self.fedn_client.set_client_id(self.client_obj.client_id)

        self.fedn_client.run()

    def set_helper(self, response: Optional[GrpcConnectionOptions] = None) -> None:
        """Set the helper based on the response or default."""
        helper_type = response.helper_type if response else None
        helper_type_to_use = self.helper_type or helper_type or "numpyhelper"
        FednLogger().info(f"Setting helper to: {helper_type_to_use}")
        self.helper = get_helper(helper_type_to_use)

    def on_train(self, in_model: BytesIO, client_settings: dict) -> Tuple[Optional[BytesIO], dict]:
        """Handle the training callback."""
        return self._process_training_request(in_model, client_settings)

    def on_validation(self, in_model: BytesIO) -> Optional[dict]:
        """Handle the validation callback."""
        return self._process_validation_request(in_model)

    def on_forward(self, client_id, is_sl_inference):
        out_embeddings, meta = self._process_forward_request(client_id, is_sl_inference)
        return out_embeddings, meta

    def on_backward(self, in_gradients, client_id):
        meta = self._process_backward_request(in_gradients, client_id)
        return meta

    def _process_training_request(self, in_model: BytesIO, client_settings: dict) -> Tuple[Optional[BytesIO], dict]:
        """Process a training (model update) request."""
        try:
            meta = {}
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(in_model.getbuffer())

            save_metadata(metadata=client_settings, filename=inpath)
            outpath = self.helper.get_tmp_path()
            tic = time.time()

            self._package_runtime.dispatcher.run_cmd(f"train {inpath} {outpath}")
            meta["exec_training"] = time.time() - tic

            with open(outpath, "rb") as fr:
                out_model = io.BytesIO(fr.read())

            with open(outpath + "-metadata", "r") as fh:
                training_metadata = json.loads(fh.read())

            FednLogger().info(f"SETTING Training metadata: {training_metadata}")
            meta["training_metadata"] = training_metadata

            os.unlink(inpath)
            os.unlink(outpath)
            os.unlink(outpath + "-metadata")

        except Exception as e:
            FednLogger().error(f"Could not process training request due to error: {e}")
            out_model = None
            meta = {"status": "failed", "error": str(e)}

        return out_model, meta

    def _process_validation_request(self, in_model: BytesIO) -> Optional[dict]:
        """Process a validation request."""
        try:
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(in_model.getbuffer())

            outpath = get_tmp_path()
            self._package_runtime.dispatcher.run_cmd(f"validate {inpath} {outpath}")

            with open(outpath, "r") as fh:
                metrics = json.loads(fh.read())

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            FednLogger().warning(f"Validation failed with exception {e}")
            metrics = None

        return metrics

    def _process_prediction_request(self, in_model: BytesIO) -> Optional[dict]:
        """Process a prediction request."""
        try:
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(in_model.getbuffer())

            outpath = get_tmp_path()
            self._package_runtime.dispatcher.run_cmd(f"predict {inpath} {outpath}")

            with open(outpath, "r") as fh:
                metrics = json.load(fh)

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            FednLogger().warning(f"Prediction failed with exception {e}")
            metrics = None

        return metrics

    def _process_forward_request(self, client_id, is_sl_inference) -> Tuple[BytesIO, dict]:
        """Process a forward request. Param is_sl_inference determines whether the forward pass is used for gradient calculation or validation.

        :param client_id: The client ID.
        :type client_id: str
        :param is_sl_inference: Whether the request is for splitlearning inference or not.
        :type is_sl_inference: str
        :return: The embeddings, or None if forward failed.
        :rtype: tuple
        """
        try:
            out_embedding_path = get_tmp_path()

            tic = time.time()
            self._package_runtime.dispatcher.run_cmd(f"forward {client_id} {out_embedding_path} {is_sl_inference}")

            meta = {}
            embeddings = None

            with open(out_embedding_path, "rb") as fr:
                embeddings = io.BytesIO(fr.read())

            meta["exec_training"] = time.time() - tic

            # Read the metadata file
            with open(out_embedding_path + "-metadata", "r") as fh:
                training_metadata = json.loads(fh.read())

            FednLogger().debug("SETTING Forward metadata: {}".format(training_metadata))
            meta["training_metadata"] = training_metadata

            os.unlink(out_embedding_path)
            os.unlink(out_embedding_path + "-metadata")

        except Exception as e:
            FednLogger().warning("Forward failed with exception {}".format(e))
            embeddings = None
            meta = {"status": "failed", "error": str(e)}

        return embeddings, meta

    def _process_backward_request(self, in_gradients: BytesIO, client_id: str) -> dict:
        """Process a backward request.

        :param in_gradients: The gradients to be processed.
        :type in_gradients: BytesIO
        :return: Metadata, or None if backward failed.
        :rtype: dict
        """
        try:
            meta = {}
            inpath = get_tmp_path()

            # load gradients
            with open(inpath, "wb") as fh:
                fh.write(in_gradients.getbuffer())

            tic = time.time()

            self._package_runtime.dispatcher.run_cmd(f"backward {inpath} {client_id}")
            meta["exec_training"] = time.time() - tic

            os.unlink(inpath)

        except Exception as e:
            FednLogger().error("Backward failed with exception {}".format(e))
            meta = {"status": "failed", "error": str(e)}

        return meta
