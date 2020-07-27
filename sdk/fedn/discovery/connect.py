import enum

import requests as r


class State(enum.Enum):
    Disconnected = 0
    Connected = 1
    Error = 2


class DiscoveryClientConnect:

    def __init__(self, host, port, token, name):
        self.host = host
        self.port = port
        self.token = token
        self.name = name
        self.state = State.Disconnected
        self.connect_string = "http://{}:{}".format(self.host, self.port)
        print("\n\nsetting the connection string to {}\n\n".format(self.connect_string), flush=True)

    def state(self):
        return self.state

    def connect(self):

        try:
            retval = r.get("{}{}/".format(self.connect_string + '/client/', self.name),
                           headers={'Authorization': 'Token {}'.format(self.token)})
        except Exception as e:
            self.state = State.Disconnected
            return self.state

        if retval.status_code != 200:

            payload = {'name': self.name, 'status': "R", 'user': 1}
            retval = r.post(self.connect_string + '/client/', data=payload,
                            headers={'Authorization': 'Token {}'.format(self.token)})
            #print("status is {} and payload {}".format(retval.status_code, retval.text), flush=True)
            if retval.status_code >= 200 or retval.status_code < 204:
                self.state = State.Connected
            else:
                self.state = State.Disconnected
        else:
            self.state = State.Connected

        return self.state

    def update_status(self, status):
        #print("\n\nUpdate status", flush=True)
        payload = {'status': status}
        retval = r.patch("{}{}/".format(self.connect_string + '/client/', self.name), data=payload,
                         headers={'Authorization': 'Token {}'.format(self.token)})

        #print("SETTING UPDATE< WHAT HAPPENS {} {}".format(retval.status_code, retval.text), flush=True)
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
        #print("Got payload {}".format(payload), flush=True)
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


class DiscoveryCombinerConnect(DiscoveryClientConnect):

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

        print("SETTING UPDATE< WHAT HAPPENS {} {}".format(retval.status_code, retval.text), flush=True)
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

    def get_config(self):
        retval = r.get("{}{}/".format(self.connect_string + '/configuration/', self.myname),
                       headers={'Authorization': 'Token {}'.format(self.token)})

        payload = retval.json()
        print("GOT CONFIG: {}".format(payload))

        return payload, self.state
