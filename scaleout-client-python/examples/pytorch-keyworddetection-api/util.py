import yaml


def construct_api_url(api_url: str, api_port: int = None, secure: bool = None) -> str:
    """Constructs a valid API URL from the input parameters."""
    api_url = api_url.strip(" ")

    if "://" in api_url:
        scheme, api_url = api_url.split("://")

        if scheme not in ["http", "https"]:
            raise Exception("API requires http(s)")
        if secure is not None and secure != (scheme == "http"):
            raise Exception("Scheme is supplied but security flag does not match scheme")
    else:
        if secure is None:
            secure = "localhost" not in api_url and "127.0.0.1" not in api_url
        scheme = "https" if secure else "http"

    if "/" in api_url:
        host, path = api_url.split("/")
        if not path.endswith("/"):
            path += "/"
    else:
        host = api_url
        path = ""

    if api_port is not None:
        if ":" in host:
            # Overriding port
            hostname, port = host.split(":")
            host = f"{hostname}:{api_port}"
        else:
            host = f"{host}:{api_port}"

    return f"{scheme}://{host}/{path}"


def read_settings(file_path: str) -> dict:
    """Reads a YAML file and returns the content as a dictionary."""
    with open(file_path, "rb") as config_file:
        return yaml.safe_load(config_file.read())
