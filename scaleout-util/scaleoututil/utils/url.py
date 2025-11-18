from typing import Optional, Union
import urllib.parse


def build_url(scheme: str, host: str, port: Optional[Union[str, int]], endpoint: str) -> str:
    """Build a URL from its components."""
    if not host:
        return ""
    if scheme:
        url = f"{scheme}://{host}"
    else:
        url = f"{host}"
    if port:
        url += f":{int(port)}"
    if endpoint is not None:
        endpoint = endpoint.lstrip("/")
    if endpoint:
        url += "/"
        url += endpoint
    return url


def assemble_endpoint_url(*parts: str, **kwargs) -> str:
    """Assemble endpoint URL from parts and query parameters.

    Args:
        *parts: Parts of the URL to be joined.
        **kwargs: Query parameters to be appended to the URL.

    Returns:
        The assembled URL as a string.

    """
    # Remove leading slashes but keep final part's leading slash if present
    if parts:
        url = "/".join(part.strip("/") for part in parts[:-1] if part)
        if len(parts) > 1:
            url += "/"
        url += parts[-1].lstrip("/")
    else:
        url = ""
    if kwargs:
        url += "?" + urllib.parse.urlencode(kwargs)
    return url


def parse_url(url: str):
    """Extract protocol, host, and port from a given URL."""
    if "://" not in url:
        url = f"http://{url}"
        no_protocol = True
    else:
        no_protocol = False

    urlparse_result = urllib.parse.urlparse(url)
    if urlparse_result.scheme not in ["http", "https"]:
        raise ValueError(f"Unsupported URL scheme: {urlparse_result.scheme}")

    if no_protocol and urlparse_result.scheme == "http":
        urlparse_result = urlparse_result._replace(scheme=None)

    protocol = urlparse_result.scheme
    host = urlparse_result.hostname
    port = urlparse_result.port
    path = urlparse_result.path.strip("/")

    return protocol, host, port, path
