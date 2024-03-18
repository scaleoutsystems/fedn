import os

CONTROLLER_DEFAULTS = {
    'protocol': 'http',
    'host': 'localhost',
    'port': 8092,
    'debug': False
}

COMBINER_DEFAULTS = {
    'discover_host': 'localhost',
    'discover_port': 8092,
    'host': 'localhost',
    'port': 12080,
    "name": "combiner",
    "max_clients": 30
}

CLIENT_DEFAULTS = {
    'discover_host': 'localhost',
    'discover_port': 8092,
}

API_VERSION = 'v1'


def get_api_url(protocol: str, host: str, port: str, endpoint: str) -> str:
    _protocol = os.environ.get('FEDN_PROTOCOL') or protocol or CONTROLLER_DEFAULTS['protocol']
    _host = os.environ.get('FEDN_HOST') or host or CONTROLLER_DEFAULTS['host']
    _port = os.environ.get('FEDN_PORT') or port or CONTROLLER_DEFAULTS['port']

    return f'{_protocol}://{_host}:{_port}/api/{API_VERSION}/{endpoint}'


def get_token(token: str) -> str:
    _token = os.environ.get("FEDN_TOKEN", token)

    if _token is None:
        return None

    scheme = os.environ.get("FEDN_AUTH_SCHEME", "Bearer")

    return f"{scheme} {_token}"
