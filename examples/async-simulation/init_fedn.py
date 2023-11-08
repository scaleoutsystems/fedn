from fedn import APIClient

DISCOVER_HOST = '3.80.63.103'
DISCOVER_PORT = 8092

client = APIClient(DISCOVER_HOST, DISCOVER_PORT)
client.set_package('package.tar.gz', 'numpyarrayhelper')
client.set_initial_model('seed.npz')
