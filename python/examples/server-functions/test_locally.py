"""Helper function to test if your server functions implementation runs correctly."""
from client.model import compile_model
from scaleout.network.combiner.hooks.serverfunctionstest import test_server_functions
from server_functions import ServerFunctions


model = compile_model()
parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
client_metadata = {"num_examples": 1}
test_server_functions(ServerFunctions, parameters_np, client_metadata, rounds=5, num_clients=20)
