from fedn import APIClient

DISCOVER_HOST = '34.207.219.112'
DISCOVER_PORT = 8092

client = APIClient(DISCOVER_HOST, DISCOVER_PORT)
client.set_package('package.tar.gz', 'numpyarrayhelper')
client.set_initial_model('seed.npz')
