import enum

import requests as r


class State(enum.Enum):
    Disconnected = 0
    Connected = 1
    Error = 2


class DiscoveryClientConnect:

    def __init__(self, host, port, token):
        self.host = host
        self.port = port
        self.token = token
        self.state = State.Disconnected

    def state(self):
        return self.state


class DiscoveryCombinerConnect(DiscoveryClientConnect):

    def __init__(self, host, port, token, myhost, myport, myname):
        super().__init__(host, port, token)
        self.connect_string = "{}:{}/combiner/".format(self.host, self.port)
        self.myhost = myhost
        self.myport = myport
        self.myname = myname

    def connect(self):

        payload = {'name': self.myname, 'port': self.myport, 'host': self.myhost, 'status': "R"}
        retval = r.post(self.connect_string, data=payload, headers={'Authorization': 'Token {}'.format(self.token)})
        if retval.status_code == 200 or retval.status_code == 201:
            self.state = State.Connected
        else:
            self.state = State.Disconnected

    def update(self, status):

        payload = ""
        retval = r.post("{}{}".format(self.connect_string, self.myname), data=payload,
                        headers={'Authorization': 'Token {}'.format(self.token)})
        if retval.status_code == 201 or retval.status_code == 200:
            self.state = State.Connected
        else:
            self.state = State.Disconnected

    def check_status(self):

        status = None

        retval = r.get("{}{}".format(self.connect_string, self.myname),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        try:
            status = payload['status']
        except Exception as e:
            print("Error getting payload {}".format(e))
            self.state = State.Error

        return status, self.state
