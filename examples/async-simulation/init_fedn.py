from fedn import APIClient

DISCOVER_HOST = '127.0.0.1'
DISCOVER_PORT = 8092

client = APIClient(DISCOVER_HOST, DISCOVER_PORT)
client.set_active_package('package.tgz', 'numpyhelper')
client.set_active_model('seed.npz')
