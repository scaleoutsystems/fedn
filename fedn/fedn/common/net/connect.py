import enum

import requests as r


class State(enum.Enum):
    Disconnected = 0
    Connected = 1
    Error = 2


class Status(enum.Enum):
    Unassigned = 0
    Assigned = 1
    TryAgain = 2
    UnAuthorized = 3
    UnMatchedConfig = 4


class ConnectorClient:
    """
    Connector for assigning client to a combiner in the FEDn network.
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

    def state(self):
        """

        :return: Connector State
        """
        return self.state

    def assign(self):
        """
        Connect client to FEDn network discovery service, ask for combiner assignment.

        :return: Tuple with assingment status, combiner connection information
        if sucessful, else None.
        :rtype: Status, json
        """
        try:
            retval = None
            if self.preferred_combiner:
                retval = r.get("{}?name={}&combiner={}".format(self.connect_string + '/assign', self.name,
                                                               self.preferred_combiner),
                               verify=self.verify,
                               allow_redirects=True,
                               headers={'Authorization': 'Token {}'.format(self.token)})
            else:
                retval = r.get("{}?name={}".format(self.connect_string + '/assign', self.name),
                               verify=self.verify,
                               allow_redirects=True,
                               headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception as e:
            print('***** {}'.format(e), flush=True)
            return Status.Unassigned, {}

        if retval.status_code == 401:
            reason = "Unauthorized connection to reducer, make sure the correct token is set"
            return Status.UnAuthorized, reason

        reducer_package = retval.json()['package']
        if reducer_package != self.package:
            reason = "Unmatched config of compute package between client and reducer.\n" +\
                "Reducer uses {} package and client uses {}.".format(
                    reducer_package, self.package)
            return Status.UnMatchedConfig, reason

        if retval.status_code >= 200 and retval.status_code < 204:
            if retval.json()['status'] == 'retry':
                if 'msg' in retval.json():
                    reason = retval.json()['msg']
                else:
                    reason = "Reducer was not ready. Try again later."

                return Status.TryAgain, reason

            return Status.Assigned, retval.json()

        return Status.Unassigned, None


class ConnectorCombiner:
    """
    Connector for annnouncing combiner to the FEDn network.
    """

    def __init__(self, host, port, myhost, fqdn, myport, token, name, secure=False, verify=False):

        self.host = host
        self.fqdn = fqdn
        self.port = port
        self.myhost = myhost
        self.myport = myport
        self.token = token
        self.name = name
        self.secure = secure
        self.verify = verify

        # for https we assume a an ingress handles permanent redirect (308)
        self.prefix = "http://"
        if port:
            self.connect_string = "{}{}:{}".format(
                self.prefix, self.host, self.port)
        else:
            self.connect_string = "{}{}".format(
                self.prefix, self.host)

        print("\n\nsetting the connection string to {}\n\n".format(
            self.connect_string), flush=True)

    def state(self):
        """

        :return: Combiner State
        """
        return self.state

    def announce(self):
        """
        Announce combiner to FEDn network via discovery service.

        :return: Tuple with announcement Status, FEDn network configuration
        if sucessful, else None.
        :rtype: Staus, json
        """
        try:
            retval = r.get("{}?name={}&address={}&fqdn={}&port={}&secure={}".format(
                self.connect_string + '/add',
                self.name,
                self.myhost,
                self.fqdn,
                self.myport,
                self.secure),
                verify=self.verify,
                headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception:
            return Status.Unassigned, {}

        if retval.status_code == 401:
            reason = "Unauthorized connection to reducer, make sure the correct token is set"
            return Status.UnAuthorized, reason

        if retval.status_code >= 200 and retval.status_code < 204:
            if retval.json()['status'] == 'retry':
                reason = "Reducer was not ready. Try again later."
                return Status.TryAgain, reason
            return Status.Assigned, retval.json()

        return Status.Unassigned, None
