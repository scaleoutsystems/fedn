from server_functions import ServerFunctions

from scaleout import APIClient

# Fetch your host address from the studio UI and add it below.
client = APIClient(host="", secure=True, verify=True)

print(client.start_session(server_functions=ServerFunctions, helper="numpyhelper"))
