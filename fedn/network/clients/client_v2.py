import uuid

from fedn.network.clients.client_api import ClientAPI, ConnectToApiResult


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

        self.connect_string = f"{api_url}:{api_port}" if api_port else {api_url}

        self.client_api: ClientAPI = ClientAPI()

    def _connect_to_api(self) -> bool:
        response, msg = self.client_api.connect_to_api(self.connect_string, self.token, self.client_obj.to_json())
        if response == ConnectToApiResult.Assigned:
            print(msg)
            return True

        print(f"Error: {msg}")
        return False

    def start(self):
        self._connect_to_api()
