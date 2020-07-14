
import requests
from scaleout.errors import AuthenticationError
import json

def login(url,username,password):
	""" Login to Studio services. """

def get_bearer_token(url, username, password):
    """ Exchange username,password for an auth token.
        TODO: extend to get creds from keyring. """
    data = {
        'username': username,
        'password': password
    }

    r = requests.post(url, data=data)

    if r.status_code == 200:
        return json.loads(r.content)['token']
    else:
        print('Authentication failed!')
        print("Requesting an authorization token failed.")
        print('Returned status code: {}'.format(r.status_code))
        print('Reason: {}'.format(r.reason))
        raise AuthenticationError

