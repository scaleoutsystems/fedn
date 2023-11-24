from fedn import APIClient

DISCOVER_HOST = '54.208.105.152'
DISCOVER_PORT = 8092

client = APIClient(DISCOVER_HOST, DISCOVER_PORT)
client.set_package('package.tar.gz', 'numpyarrayhelper')
client.set_initial_model('seed.npz')
