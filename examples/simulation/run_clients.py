#!./.simulation/bin/python
import os

from fedn.network.clients import Client

API_SERVER = 'localhost:8092'


config = {'remote_compute_context': False,
          'validator': False,
          'trainer': True,
          'init': 'client.yaml',
          }

client = Client(config)
client.run()
