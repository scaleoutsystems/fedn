Using Flower clients in FEDn
============================

This example shows how to run a Flower 'ClientApp' using the FEDn server-side infrastructure.

See `flwr_client.py` and `flwr_task.py` for the Flower client code. The FEDn compute package is complemented
with an adapter for the Flower client, `client_app_adapter.py`.


Running the example
-------------------

See `https://fedn.readthedocs.io/en/stable/quickstart.html` for a general introduction to FEDn. This example follows the same structure
as the pytorch quickstart example. To build a virtual environment, the compute package and the seed model: 

.. code-block::

   bin/init_venv.sh

.. code-block::

   bin/build.sh

In a separate terminal, navigate to this folder, then start a client and inject the `CLIENT_NUMBER` 
dependency, for example for client1:


If you are using a FEDn Studio project:
---------------------------------------

- Register a client in Studio and obtain the corresponding 'client.yaml' 

Then start the client: 

Activate the virtual environment:
.. code-block::
   source .flower-example/bin/activate

Start the fedn client: 

.. code-block::
   CLIENT_NUMBER=0 FEDN_AUTH_SCHEME=Bearer fedn run client -in client.yaml --force-ssl --secure=True




If you prefer to use Docker:

Build an image containing the project dependencies (this might take a long time):

.. code-block::

   docker build -t flower-client .

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   -e CLIENT_NUMBER=0 \
   flower-client run client -in client.yaml --secure=True --force-ssl


If you are running FEDn locally using the provided docker-compose template:
---------------------------------------------------------------------------

Use the FEDn API Client to initalize FEDn with the compute package and seed model: 

.. code-block::

   python init_fedn.py

Create a file 'client.yaml' with the following content: 

.. code-block::
   
   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092

The start the client

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   --network=fedn_default \
   -e CLIENT_NUMBER=0 \
   flower-client run client -in client.yaml --name client1
