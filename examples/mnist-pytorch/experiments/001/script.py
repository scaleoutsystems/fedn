# Import packages
from fedn import APIClient
import time
import uuid
import json
import matplotlib.pyplot as plt
import numpy as np
import collections

# Create an API Client
DISCOVER_HOST = '127.0.0.1'
DISCOVER_PORT = 8092
client = APIClient(DISCOVER_HOST, DISCOVER_PORT)

# Set package
client.set_package('package.tgz', 'numpyhelper')

# Set intial seed
client.set_initial_model('seed.npz')
seed_model = client.get_initial_model()

# Show running client details
print("Running clients:")
for client_container in client.list_clients()['result']:
    if client_container['status'] == 'online':
        print(f"{client_container['ip']}: {client_container['id']}")

session_id = input("Enter Session ID: ")

session_config = {
    "helper": "numpyhelper",
    "session_id": session_id,
    "aggregator": "fedavg",
    "model_id": seed_model['model_id'],
    "rounds": 10
}

# Start the training session
start_result = client.start_session(**session_config)
print(f"{start_result['message']}\n")

# Sleep and check learning status
while not client.session_is_finished(session_id):
    if client.list_models(session_id):
        print(f"{client.list_models(session_id)['count']} of {session_config['rounds']} rounds completed")
    time.sleep(10)
