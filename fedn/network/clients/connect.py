# This file contains the Connector class for assigning client to the FEDn network via the discovery service (REST-API).
# The Connector class is used by the Client class in fedn/network/clients/client.py.
# Once assigned, the client will retrieve combiner assignment from the discovery service.
# The discovery service will also add the client to the statestore.
#
#
import enum

import requests

from fedn.common.config import FEDN_AUTH_REFRESH_TOKEN, FEDN_AUTH_REFRESH_TOKEN_URI, FEDN_AUTH_SCHEME, FEDN_CUSTOM_URL_PREFIX
from fedn.common.log_config import logger


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
    :type combiner: str
    :param id: id of client
    """

    def __init__(self, host, port, token, name, remote_package, force_ssl=False, verify=False, combiner=None, id=None):
        self.host = host
        self.port = port
        self.token = token
        self.name = name
        self.verify = verify
        self.preferred_combiner = combiner
        self.id = id
        self.package = "remote" if remote_package else "local"

        # for https we assume a an ingress handles permanent redirect (308)
        if force_ssl:
            self.prefix = "https://"
        else:
            self.prefix = "http://"
        if self.port:
            self.connect_string = "{}{}:{}".format(self.prefix, self.host, self.port)
        else:
            self.connect_string = "{}{}".format(self.prefix, self.host)

        logger.info("Setting connection string to {}.".format(self.connect_string))

    def assign(self):
        """Connect client to FEDn network discovery service, ask for combiner assignment.

        :return: Tuple with assingment status, combiner connection information if sucessful, else None.
        :rtype: tuple(:class:`fedn.network.clients.connect.Status`, str)
        """
        try:
            retval = None
            payload = {"name": self.name, "client_id": self.id, "preferred_combiner": self.preferred_combiner, "package": self.package}
            retval = requests.post(
                self.connect_string + FEDN_CUSTOM_URL_PREFIX + "/add_client",
                json=payload,
                verify=self.verify,
                allow_redirects=True,
                headers={"Authorization": f"{FEDN_AUTH_SCHEME} {self.token}"},
            )
        except Exception as e:
            logger.debug("***** {}".format(e))
            return Status.Unassigned, {}

        if retval.status_code == 400:
            # Get error messange from response
            reason = retval.json()["message"]
            return Status.UnMatchedConfig, reason

        if retval.status_code == 401:
            if "message" in retval.json():
                reason = retval.json()["message"]
                logger.warning(reason)
                if reason == "Token expired":
                    status_code = self.refresh_token()
                    if status_code >= 200 and status_code < 204:
                        logger.info("Token refreshed.")
                        return Status.TryAgain, reason
                    else:
                        return Status.UnAuthorized, "Could not refresh token"
            reason = "Unauthorized connection to reducer, make sure the correct token is set"
            return Status.UnAuthorized, reason

        if retval.status_code >= 200 and retval.status_code < 204:
            if retval.json()["status"] == "retry":
                if "message" in retval.json():
                    reason = retval.json()["message"]
                else:
                    reason = "Controller was not ready. Try again later."

                return Status.TryAgain, reason

            return Status.Assigned, retval.json()

        return Status.Unassigned, None

    def refresh_token(self):
        """Refresh client token.

        :return: Tuple with assingment status, combiner connection information if sucessful, else None.
        :rtype: tuple(:class:`fedn.network.clients.connect.Status`, str)
        """
        if not FEDN_AUTH_REFRESH_TOKEN_URI or not FEDN_AUTH_REFRESH_TOKEN:
            logger.error("No refresh token URI/Token set, cannot refresh token.")
            return 401

        payload = requests.post(FEDN_AUTH_REFRESH_TOKEN_URI, verify=self.verify, allow_redirects=True, json={"refresh": FEDN_AUTH_REFRESH_TOKEN})
        self.token = payload.json()["access"]
        return payload.status_code
