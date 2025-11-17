from scaleout import APIClient

DISCOVER_HOST = "127.0.0.1"
DISCOVER_PORT = 8092

client = APIClient(DISCOVER_HOST, DISCOVER_PORT)
client.set_package("package.tgz", "numpyhelper")
client.set_initial_model("seed.npz")
