import uuid
from typing import Tuple

from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_CUSTOM_URL_PREFIX
from fedn.common.log_config import logger
from fedn.network.clients.client_api import ClientAPI, ConnectToApiResult


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
    def __init__(self, api_url: str, api_port: int, client_obj: ClientOptions, token: str = None):
        self.api_url = api_url
        self.api_port = api_port
        self.token = token
        self.client_obj = client_obj

        self.connect_string = get_url(self.api_url, self.api_port)
        logger.info(self.connect_string)

        self.client_api: ClientAPI = ClientAPI()

    def _connect_to_api(self) -> Tuple[bool, dict]:
        result, response = self.client_api.connect_to_api(self.connect_string, self.token, self.client_obj.to_json())
        if result == ConnectToApiResult.Assigned:
            return True, response

        logger.error(f"Error: {response}")
        return False, None

    def _download_compute_package(self):
        pass

    def start(self):
        result, response = self._connect_to_api()

        if not result:
            return

        logger.info("Client assinged to controller")

        if self.client_obj.package == "remote":
            result: bool = self._download_compute_package()
            if not result:
                logger.error("Could not download compute package")
                return
            result: bool = self.client_api.set_compute_package_checksum(self.connect_string, self.token)
            if not result:
                logger.error("Could not set checksum")
                return

            # TODO: validate package (add checksum)
            self.client_api.unpack_compute_package()
