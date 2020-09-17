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


from fedn.common.security.certificate import Certificate


class ConnectorClient:

    def __init__(self, host, port, token, name, id=None, secure=True, preshared_cert=True, verify_cert=False):
        self.host = host
        self.port = port
        self.token = token
        self.name = name
        self.id = id
        self.verify_cert = verify_cert
        #        self.state = State.Disconnected
        self.secure = secure
        if not secure:
            prefix = "http://"
        else:
            prefix = "https://"
        if secure and preshared_cert:
            import os
            self.certificate = Certificate(os.getcwd() + "/certs/", name="client", key_name="client-key.pem", cert_name="client-cert.pem").cert_path
        else:
            self.verify_cert = False
        self.prefix = prefix
        self.connect_string = "{}{}:{}".format(self.prefix, self.host, self.port)
        print("\n\nsetting the connection string to {}\n\n".format(self.connect_string), flush=True)
        print("Securely connecting with certificate {}".format(self.certificate), flush=True)

    def state(self):
        return self.state

    def assign(self):

        try:
            retval = r.get("{}?name={}".format(self.connect_string + '/assign', self.name), verify=str(self.certificate),
                           headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception as e:
            # self.state = State.Disconnected
            return Status.Unassigned, {}

        if retval.status_code >= 200 and retval.status_code < 204:
            print("CLIENT: client assign request was successful, returning json payload {}".format(retval.json()),
                  flush=True)
            return Status.Assigned, retval.json()

        return Status.Unassigned, None


class ConnectorCombiner:

    def __init__(self, host, port, myhost, myport, token, name, secure=True, preshared_cert=True, verify_cert=False):
        self.host = host
        self.port = port
        self.myhost = myhost
        self.myport = myport
        self.token = token
        self.name = name
        # self.state = State.Disconnected
        self.secure = secure
        if not secure:
            prefix = "http://"
        else:
            prefix = "https://"
        if secure and preshared_cert:
            import os
            self.certificate = Certificate(os.getcwd() + "/certs/", name="client", key_name="client-key.pem", cert_name="client-cert.pem",
                                           ).cert_path
        else:
            self.verify_cert = False
        self.prefix = prefix

        self.connect_string = "{}{}:{}".format(self.prefix, self.host, self.port)
        print("\n\nsetting the connection string to {}\n\n".format(self.connect_string), flush=True)
        print("Securely connecting with certificate {}".format(self.certificate), flush=True)

    def state(self):
        return self.state

    def announce(self):

        try:
            retval = r.get("{}?name={}&address={}&port={}".format(self.connect_string + '/add',
                                                                  self.name,
                                                                  self.myhost,
                                                                  self.myport),
                           verify=str(self.certificate),
                           headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception as e:
            # self.state = State.Disconnected
            return Status.Unassigned, {}

        if retval.status_code >= 200 and retval.status_code < 204:
            #print("CLIENT: client assign request was successful, returning json payload {}".format(retval.json()),
            #      flush=True)
            return Status.Assigned, retval.json()

        return Status.Unassigned, None


"""
    def connect(self):

        try:
            retval = r.get("{}{}/".format(self.connect_string + '/client/', self.name),
                           headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception as e:
            self.state = State.Disconnected
            return self.state

        if retval.status_code != 200:

            payload = {'name': self.name, 'status': "R", 'user': self.id}
            retval = r.post(self.connect_string + '/client/', data=payload,
                            headers={'Authorization': 'Token {}'.format(self.token)})
            # print("status is {} and payload {}".format(retval.status_code, retval.text), flush=True)
            if retval.status_code >= 200 or retval.status_code < 204:
                self.state = State.Connected
            else:
                self.state = State.Disconnected
        else:
            self.state = State.Connected

        return self.state

    def update_status(self, status):
        # print("\n\nUpdate status", flush=True)
        payload = {'status': status}
        retval = r.patch("{}{}/".format(self.connect_string + '/client/', self.name), data=payload,
                         headers={'Authorization': 'Token {}'.format(self.token)})

        # print("SETTING UPDATE< WHAT HAPPENS {} {}".format(retval.status_code, retval.text), flush=True)
        if retval.status_code >= 200 or retval.status_code < 204:
            self.state = State.Connected
        else:
            self.state = State.Disconnected

        newstatus = None

        retval = r.get("{}{}/".format(self.connect_string + '/client/', self.name),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        try:
            newstatus = payload['status']
        except Exception as e:
            print("Error getting payload {}".format(e))
            self.state = State.Error

        return newstatus

    def check_status(self):
        print("\n\nCheck status", flush=True)
        status = None

        retval = r.get("{}{}/".format(self.connect_string + '/client/', self.name),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        # print("Got payload {}".format(payload), flush=True)
        try:
            status = payload['status']
        except Exception as e:
            print("Error getting payload {}".format(e))
            self.state = State.Error

        return status, self.state

    def get_config(self):
        retval = r.get("{}{}/".format(self.connect_string + '/client/', self.name),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        print("GOT CONFIG: {}".format(payload))
        retval = r.get("{}?id={}".format(self.connect_string + '/combiner/', payload['combiner']),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        combiner_payload = retval.json()[0]
        print("GOT HOST PORT ETC SET TO {}".format(combiner_payload))

        return combiner_payload, self.state


"""
"""
class DiscoveryCombinerConnect(ConnectorClient):

    def __init__(self, host, port, token, myhost, myport, myname):
        super().__init__(host, port, token, myname)
        self.connect_string = "http://{}:{}".format(self.host, self.port)
        self.myhost = myhost
        self.myport = myport
        self.myname = myname
        print("\n\nsetting the connection string to {}\n\n".format(self.connect_string), flush=True)

    def connect(self):

        retval = r.get("{}{}/".format(self.connect_string + '/combiner/', self.myname),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        if 200 <= retval.status_code < 204:
            if retval.json()['status'] != 'R':
                print("Recovering from previous state. Resetting to Ready\n\n", flush=True)
                status = 'R'
                payload = {'status': status}
                retval = r.patch("{}{}/".format(self.connect_string + '/combiner/', self.myname), data=payload,
                                 headers={'Authorization': 'Token {}'.format(self.token)})

        if retval.status_code != 200:

            # print("Got payload {}".format(ret), flush=True)
            payload = {'name': self.myname, 'port': self.myport, 'host': self.myhost, 'status': "S", 'user': 1}
            retval = r.post(self.connect_string + '/combiner/', data=payload,
                            headers={'Authorization': 'Token {}'.format(self.token)})
            print("status is {} and payload {}".format(retval.status_code, retval.text), flush=True)
            if 200 <= retval.status_code < 204:
                self.state = State.Connected
            else:
                self.state = State.Disconnected
        else:
            self.state = State.Connected

        return self.state

    def update_status(self, status):
        print("\n\nUpdate status", flush=True)
        payload = {'status': status}
        retval = r.patch("{}{}/".format(self.connect_string + '/combiner/', self.myname), data=payload,
                         headers={'Authorization': 'Token {}'.format(self.token)})

        # print("SETTING UPDATE< WHAT HAPPENS {} {}".format(retval.status_code, retval.text), flush=True)
        if 200 <= retval.status_code < 204:
            self.state = State.Connected
        else:
            self.state = State.Disconnected

        newstatus = None

        retval = r.get("{}{}/".format(self.connect_string + '/combiner/', self.myname),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        try:
            newstatus = payload['status']
        except Exception as e:
            print("Error getting payload {}".format(e))
            self.state = State.Error

        return newstatus

    def check_status(self):
        print("\n\nCheck status", flush=True)
        status = None

        retval = r.get("{}{}/".format(self.connect_string + '/combiner/', self.myname),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        print("Got payload {}".format(payload), flush=True)
        try:
            status = payload['status']
        except Exception as e:
            print("Error getting payload {}".format(e))
            self.state = State.Error

        return status, self.state

    def get_combiner_config(self):
        retval = r.get("{}{}/".format(self.connect_string + '/combiner/', self.myname),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        print("GOT CONFIG: {}".format(payload))

        return payload, self.state

    def get_config(self):
        retval = r.get("{}{}/".format(self.connect_string + '/configuration/', self.myname),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        print("GOT CONFIG: {}".format(payload))

        return payload, self.state
"""
