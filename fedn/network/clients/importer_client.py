"""Client module for handling client operations in the FEDn network."""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from fedn.common.config import FEDN_CUSTOM_URL_PREFIX
from fedn.common.log_config import logger
from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient, GrpcConnectionOptions
from fedn.network.clients.importer_package_runtime import ImporterPackageRuntime, get_compute_package_dir_path
from fedn.utils.process import _IS_UNIX, _join_commands


def get_url(api_url: str, api_port: int) -> str:
    """Construct the URL for the API."""
    return f"{api_url}:{api_port}/{FEDN_CUSTOM_URL_PREFIX}" if api_port else f"{api_url}/{FEDN_CUSTOM_URL_PREFIX}"


class ClientOptions:
    """Options for configuring the client."""

    def __init__(self, name: str, package: str, preferred_combiner: Optional[str] = None, client_id: Optional[str] = None) -> None:
        """Initialize ClientOptions with validation."""
        self._validate(name, package)
        self.name = name
        self.package = package
        self.preferred_combiner = preferred_combiner
        self.client_id = client_id if client_id else str(uuid.uuid4())

    def _validate(self, name: str, package: str) -> None:
        """Validate the name and package."""
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("Name must be a string")
        if not isinstance(package, str) or len(package) == 0 or package not in ["local", "remote"]:
            raise ValueError("Package must be either 'local' or 'remote'")

    def to_json(self) -> Dict[str, Optional[str]]:
        """Convert ClientOptions to JSON."""
        return {
            "name": self.name,
            "client_id": self.client_id,
            "preferred_combiner": self.preferred_combiner,
            "package": self.package,
        }


class ImporterClient:
    """Client for interacting with the FEDn network."""

    def __init__(
        self,
        api_url: str,
        api_port: int,
        client_obj: ClientOptions,
        combiner_host: Optional[str] = None,
        combiner_port: Optional[int] = None,
        token: Optional[str] = None,
        package_checksum: Optional[str] = None,
        helper_type: Optional[str] = None,
        startup_path: Optional[str] = None,
        managed_env: Optional[bool] = True,
    ) -> None:
        """Initialize the Client."""
        self.api_url = api_url
        self.api_port = api_port
        self.combiner_host = combiner_host
        self.combiner_port = combiner_port
        self.token = token
        self.client_obj = client_obj
        self.package_checksum = package_checksum
        self.helper_type = helper_type

        package_path, archive_path = get_compute_package_dir_path()
        self.package_runtime = ImporterPackageRuntime(package_path, archive_path)

        self.managed_env = managed_env

        self.fedn_api_url = get_url(self.api_url, self.api_port)
        self.fedn_client: FednClient = FednClient()
        self.helper = None
        self.startup_path = startup_path

    def _connect_to_api(self) -> Tuple[bool, Optional[dict]]:
        """Connect to the API and handle retries."""
        result = None
        response = None

        while not result or result == ConnectToApiResult.ComputePackageMissing:
            if result == ConnectToApiResult.ComputePackageMissing:
                logger.info("Retrying in 3 seconds")
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
            result = self.package_runtime.load_remote_compute_package(url=self.fedn_api_url, token=self.token, package_checksum=self.package_checksum)
            if not result:
                return
        else:
            result = self.package_runtime.load_local_compute_package(os.path.join(os.getcwd(), "client"))
            if not result:
                return

        result = self.fedn_client.init_grpchandler(config=combiner_config, client_name=self.client_obj.client_id, token=self.token)
        if not result:
            return

        self.fedn_client.set_name(self.client_obj.name)
        self.fedn_client.set_client_id(self.client_obj.client_id)

        if self.managed_env:
            logger.info("Initializing managed environment.")
            self.package_runtime.init_env_runtime()

            if not self.verify_active_environment():
                self.__restart_client_with_env()
            else:
                logger.info("Managed environment is active and verified.")

        self.package_runtime.run_startup(self.fedn_client)

        self.fedn_client.run()

    def verify_active_environment(self) -> None:
        """Verify the Python environment."""
        if self.package_runtime is None or self.package_runtime.python_env is None:
            logger.error("Package runtime or Python environment is not initialized.")
            return False
        if self.managed_env:
            venv_path = os.environ.get("VIRTUAL_ENV")
            if not venv_path:
                logger.warning("No virtual environment detected.")
                return False
            logger.info(f"Virtual environment detected at: {venv_path}")
            if Path(venv_path) != Path(self.package_runtime.python_env.path):
                logger.warning(f"Virtual environment path {venv_path} does not match the expected path {self.package_runtime.python_env.path}.")
                return False
            return True
        else:
            logger.info("Managed environment is disabled, skipping verification.")
            return True

    def __restart_client_with_env(self) -> None:
        """Restart the client."""
        # This method could be replace by letting a process manager handle the restart, i.e. a watchdog or supervisor.
        # The watchdog would monitor the client process and restart it if it exits unexpectedly
        # and start the client with the correct environment activated.
        logger.info("Restarting client with managed environment.")
        args = " ".join(sys.argv)
        logger.info(f"Current command line arguments: {args}")
        args_after_start = args.split("client start", 1)[1].strip() if "client start" in args else ""

        # TODO: Maybe we need to close open connections or clean up resources before restarting.

        activate_env_cmd = self.package_runtime.python_env.get_activate_cmd()
        cmd = _join_commands(activate_env_cmd, "python -m fedn client start " + args_after_start)
        logger.info(f"Restarting with cmd: {cmd}")
        time.sleep(2)
        entry_point = "/bin/bash" if _IS_UNIX else "C:\\Windows\\System32\\cmd.exe"
        os.execv(entry_point, cmd)  # noqa: S606
        # This line will never be reached, as os.execv replaces the current process with a new one.
