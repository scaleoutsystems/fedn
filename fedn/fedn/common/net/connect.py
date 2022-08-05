import enum
import os

import requests as r
import urllib3
from fedn.common.security.certificate import Certificate


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

    def __init__(self, host, port, token, name, remote_package, combiner=None, id=None, secure=False, preshared_cert=True,
                 verify_cert=False):

        if not verify_cert:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.host = host
        self.port = port
        self.token = token
        self.name = name
        self.preferred_combiner = combiner
        self.id = id
        self.verify_cert = verify_cert
        self.secure = secure
        self.certificate = None
        self.package = 'remote' if remote_package else 'local'

        if not secure:
            self.prefix = "http://"
        else:
            self.prefix = "https://"

        if secure and preshared_cert:
            self.certificate = Certificate(os.getcwd() + "/certs/", name="client", key_name="client-key.pem",
                                           cert_name="client-cert.pem").cert_path
        else:
            self.verify_cert = False

        self.connect_string = "{}{}:{}".format(
            self.prefix, self.host, self.port)

        print("\n\nsetting the connection string to {}\n\n".format(
            self.connect_string), flush=True)

    def state(self):
        """

        :return: Connector State
        """
        return self.state

    def assign(self):
        """
        Connect client to FEDn network via discovery service, ask for combiner assignment.

        :return: Tuple with assingment status, combiner connection information
        if sucessful, else None.
        :rtype: Status, json
        """
        try:
            cert = str(self.certificate) if self.verify_cert else False
            retval = None
            if self.preferred_combiner:
                retval = r.get("{}?name={}&combiner={}".format(self.connect_string + '/assign', self.name,
                                                               self.preferred_combiner), verify=cert,
                               headers={'Authorization': 'Token {}'.format(self.token)})
            else:
                retval = r.get("{}?name={}".format(self.connect_string + '/assign', self.name), verify=cert,
                               headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception as e:
            print('***** {}'.format(e), flush=True)
            # self.state = State.Disconnected
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

    def __init__(self, host, port, myhost, fqdn, myport, token, name, secure=False, preshared_cert=True, verify_cert=False):

        if not verify_cert:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.host = host
        self.fqdn = fqdn
        self.port = port
        self.myhost = myhost
        self.myport = myport
        self.token = token
        self.name = name
        self.verify_cert = verify_cert
        self.secure = secure

        # if not secure:
        self.prefix = "http://"
        # else:
        #    self.prefix = "https://"

        if secure and preshared_cert:
            self.certificate = Certificate(os.getcwd() + "/certs/", name="client", key_name="client-key.pem",
                                           cert_name="client-cert.pem",
                                           ).cert_path
        else:
            self.verify_cert = False

        self.connect_string = "{}{}:{}".format(
            self.prefix, self.host, self.port)

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
            cert = str(self.certificate) if self.verify_cert else False
            retval = r.get("{}?name={}&address={}&fqdn={}&port={}".format(
                self.connect_string + '/add',
                self.name,
                self.myhost,
                self.fqdn,
                self.myport),
                verify=cert,
                headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception:
            # self.state = State.Disconnected
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
