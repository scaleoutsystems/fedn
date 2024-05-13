# This file contains the Connector class for announcing combiner to the FEDn network via the discovery service (REST-API).
# The Connector class is used by the Combiner class in fedn/network/combiner/server.py.
# Once announced, the combiner will be able to receive controller requests from the controllerStub via gRPC.
# The discovery service will also add the combiner to the statestore.
#
#
import enum
import os

import requests

from fedn.common.log_config import logger


class Status(enum.Enum):
    """Enum for representing the status of a combiner announcement."""

    Unassigned = 0
    Assigned = 1
    TryAgain = 2
    UnAuthorized = 3
    UnMatchedConfig = 4


class ConnectorCombiner:
    """Connector for annnouncing combiner to the FEDn network.

    :param host: host of discovery service
    :type host: str
    :param port: port of discovery service
    :type port: int
    :param myhost: host of combiner
    :type myhost: str
    :param fqdn: fully qualified domain name of combiner
    :type fqdn: str
    :param myport: port of combiner
    :type myport: int
    :param token: token for authentication
    :type token: str
    :param name: name of combiner
    :type name: str
    :param secure: True if https is used, False if http
    :type secure: bool
    :param verify: True if certificate is verified, False if not
    :type verify: bool
    """

    def __init__(self, host, port, myhost, fqdn, myport, token, name, secure=False, verify=False):
        """Initialize the ConnectorCombiner.

        :param host: The host of the discovery service.
        :type host: str
        :param port: The port of the discovery service.
        :type port: int
        :param myhost: The host of the combiner.
        :type myhost: str
        :param fqdn: The fully qualified domain name of the combiner.
        :type fqdn: str
        :param myport: The port of the combiner.
        :type myport: int
        :param token: The token for the discovery service.
        :type token: str
        :param name: The name of the combiner.
        :type name: str
        :param secure: Use https for the connection to the discovery service.
        :type secure: bool
        :param verify: Verify the connection to the discovery service.
        :type verify: bool
        """
        self.host = host
        self.fqdn = fqdn
        self.port = port
        self.myhost = myhost
        self.myport = myport
        self.token = token
        self.token_scheme = os.environ.get("FEDN_AUTH_SCHEME", "Bearer")
        self.name = name
        self.secure = secure
        self.verify = verify

        if not self.token:
            self.token = os.environ.get("FEDN_AUTH_TOKEN", None)

        # for https we assume a an ingress handles permanent redirect (308)
        self.prefix = "http://"
        if port:
            self.connect_string = "{}{}:{}".format(self.prefix, self.host, self.port)
        else:
            self.connect_string = "{}{}".format(self.prefix, self.host)

        logger.info("Setting connection string to {}".format(self.connect_string))

    def announce(self):
        """Announce combiner to FEDn network via discovery service (REST-API).

        :return: Tuple with announcement Status, FEDn network configuration if sucessful, else None.
        :rtype: :class:`fedn.network.combiner.connect.Status`, str
        """
        payload = {"combiner_id": self.name, "address": self.myhost, "fqdn": self.fqdn, "port": self.myport, "secure_grpc": self.secure}
        url_prefix = os.environ.get("FEDN_CUSTOM_URL_PREFIX", "")
        try:
            retval = requests.post(
                self.connect_string + url_prefix + "/add_combiner",
                json=payload,
                verify=self.verify,
                headers={"Authorization": f"{self.token_scheme} {self.token}"},
            )
        except Exception:
            return Status.Unassigned, {}

        if retval.status_code == 400:
            # Get error messange from response
            reason = retval.json()["message"]
            return Status.UnMatchedConfig, reason

        if retval.status_code == 401:
            reason = "Unauthorized connection to reducer, make sure the correct token is set"
            return Status.UnAuthorized, reason

        if retval.status_code >= 200 and retval.status_code < 204:
            if retval.json()["status"] == "retry":
                reason = retval.json()["message"]
                return Status.TryAgain, reason
            return Status.Assigned, retval.json()

        return Status.Unassigned, None
