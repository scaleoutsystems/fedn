"""Connector class for assigning clients to the FEDn network via the discovery service (REST-API).

The Connector class is used by the Client class in fedn/network/clients/client.py.
Once assigned, the client will retrieve combiner assignment from the discovery service.
The discovery service will also add the client to the statestore.
"""

import enum
from typing import Optional, Tuple

import requests

from fedn.common.config import (
    FEDN_AUTH_REFRESH_TOKEN,
    FEDN_AUTH_REFRESH_TOKEN_URI,
    FEDN_AUTH_SCHEME,
    FEDN_CUSTOM_URL_PREFIX,
)
from fedn.common.log_config import logger

# Constants for HTTP status codes
HTTP_STATUS_OK = 200
HTTP_STATUS_NO_CONTENT = 204
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


class Status(enum.Enum):
    """Enum for representing the status of a client assignment."""

    Unassigned = 0
    Assigned = 1
    TryAgain = 2
    UnAuthorized = 3
    UnMatchedConfig = 4


class ConnectorClient:
    """Connector for assigning client to a combiner in the FEDn network.

    :param host: host of discovery service
    :type host: str
    :param port: port of discovery service
    :type port: int
    :param token: token for authentication
    :type token: str
    :param name: name of client
    :type name: str
    :param remote_package: True if remote package is used, False if local
    :type remote_package: bool
    :param force_ssl: True if https is used, False if http
    :type force_ssl: bool
    :param verify: True if certificate is verified, False if not
    :type verify: bool
    :param combiner: name of preferred combiner
    :type combiner: Optional[str]
    :param id: id of client
    :type id: Optional[str]
    """

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        name: str,
        remote_package: bool,
        force_ssl: bool = False,
        verify: bool = False,
        combiner: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """Initialize the ConnectorClient."""
        self.host = host
        self.port = port
        self.token = token
        self.name = name
        self.verify = verify
        self.preferred_combiner = combiner
        self.id = id
        self.package = "remote" if remote_package else "local"

        # for https we assume an ingress handles permanent redirect (308)
        self.prefix = "https://" if force_ssl else "http://"
        self.connect_string = f"{self.prefix}{self.host}:{self.port}" if self.port else f"{self.prefix}{self.host}"

        logger.info(f"Setting connection string to {self.connect_string}.")

    def assign(self) -> Tuple[Status, Optional[dict]]:
        """Connect client to FEDn network discovery service, ask for combiner assignment.

        :return: Tuple with assignment status, combiner connection information if successful, else None.
        :rtype: tuple(:class:`fedn.network.clients.connect.Status`, Optional[dict])
        """
        try:
            payload = {
                "name": self.name,
                "client_id": self.id,
                "preferred_combiner": self.preferred_combiner,
                "package": self.package,
            }
            retval = requests.post(
                self.connect_string + FEDN_CUSTOM_URL_PREFIX + "/add_client",
                json=payload,
                verify=self.verify,
                allow_redirects=True,
                headers={"Authorization": f"{FEDN_AUTH_SCHEME} {self.token}"},
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as e:
            logger.debug(f"***** {e}")
            return Status.Unassigned, {}

        if retval.status_code == HTTP_STATUS_BAD_REQUEST:
            reason = retval.json()["message"]
            return Status.UnMatchedConfig, reason

        if retval.status_code == HTTP_STATUS_UNAUTHORIZED:
            reason = retval.json().get("message", "Unauthorized connection to reducer, make sure the correct token is set")
            logger.warning(reason)
            if reason == "Token expired":
                status_code = self.refresh_token()
                if HTTP_STATUS_OK <= status_code < HTTP_STATUS_NO_CONTENT:
                    logger.info("Token refreshed.")
                    return Status.TryAgain, reason
                return Status.UnAuthorized, "Could not refresh token"
            return Status.UnAuthorized, reason

        if HTTP_STATUS_OK <= retval.status_code < HTTP_STATUS_NO_CONTENT:
            if retval.json().get("status") == "retry":
                reason = retval.json().get("message", "Controller was not ready. Try again later.")
                return Status.TryAgain, reason

            return Status.Assigned, retval.json()

        return Status.Unassigned, None

    def refresh_token(self) -> int:
        """Refresh client token.

        :return: Status code of the token refresh request.
        :rtype: int
        """
        if not FEDN_AUTH_REFRESH_TOKEN_URI or not FEDN_AUTH_REFRESH_TOKEN:
            logger.error("No refresh token URI/Token set, cannot refresh token.")
            return HTTP_STATUS_UNAUTHORIZED

        payload = requests.post(
            FEDN_AUTH_REFRESH_TOKEN_URI,
            verify=self.verify,
            allow_redirects=True,
            json={"refresh": FEDN_AUTH_REFRESH_TOKEN},
            timeout=REQUEST_TIMEOUT,
        )
        self.token = payload.json()["access"]
        return payload.status_code
