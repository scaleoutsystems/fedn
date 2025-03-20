from server_functions import ServerFunctions

from fedn import APIClient

client = APIClient(host="", secure=True, verify=True)

print(client.start_session(server_functions=ServerFunctions, helper="numpyhelper"))
