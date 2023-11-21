# This file contains the Connector class for assigning client to the FEDn network via the discovery service (REST-API).
# The Connector class is used by the Client class in fedn/network/clients/client.py.
# Once assigned, the client will retrieve combiner assignment from the discovery service.
# The discovery service will also add the client to the statestore.
#
#
import enum

import requests


class Status(enum.Enum):
    """ Enum for representing the status of a client assignment."""
    Unassigned = 0
    Assigned = 1
    TryAgain = 2
    UnAuthorized = 3
    UnMatchedConfig = 4


class ConnectorClient:
    """ Connector for assigning client to a combiner in the FEDn network.

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
        self.package = 'remote' if remote_package else 'local'

        # for https we assume a an ingress handles permanent redirect (308)
        if force_ssl:
            self.prefix = "https://"
        else:
            self.prefix = "http://"
        if self.port:
            self.connect_string = "{}{}:{}".format(
                self.prefix, self.host, self.port)
        else:
            self.connect_string = "{}{}".format(
                self.prefix, self.host)

        print("\n\nsetting the connection string to {}\n\n".format(
            self.connect_string), flush=True)

    def assign(self):
        """
        Connect client to FEDn network discovery service, ask for combiner assignment.

        :return: Tuple with assingment status, combiner connection information if sucessful, else None.
        :rtype: tuple(:class:`fedn.network.clients.connect.Status`, str)
        """
        try:
            retval = None
            payload = {'client_id': self.name, 'preferred_combiner': self.preferred_combiner}

            retval = requests.post(self.connect_string + '/add_client',
                                   json=payload,
                                   verify=self.verify,
                                   allow_redirects=True,
                                   headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception as e:
            print('***** {}'.format(e), flush=True)
            return Status.Unassigned, {}

        if retval.status_code == 400:
            # Get error messange from response
            reason = retval.json()['message']
            return Status.UnMatchedConfig, reason

        if retval.status_code == 401:
            reason = "Unauthorized connection to reducer, make sure the correct token is set"
            return Status.UnAuthorized, reason

        if retval.status_code >= 200 and retval.status_code < 204:
            if retval.json()['status'] == 'retry':
                if 'message' in retval.json():
                    reason = retval.json()['message']
                else:
                    reason = "Reducer was not ready. Try again later."

                return Status.TryAgain, reason

            reducer_package = retval.json()['package']
            if reducer_package != self.package:
                reason = "Unmatched config of compute package between client and reducer.\n" +\
                    "Reducer uses {} package and client uses {}.".format(
                        reducer_package, self.package)
                return Status.UnMatchedConfig, reason

            return Status.Assigned, retval.json()

        return Status.Unassigned, None
