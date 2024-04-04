Using Flower clients in FEDn
============================

This example demonstrates how to run a Flower 'ClientApp' on FEDn.

The FEDn compute package 'client/entrypoint' 
uses a built-in Flower compatibiltiy adapter for convenient wrapping of the Flower client.
See `flwr_client.py` and `flwr_task.py` for the Flower client code (which is adapted from 
https://github.com/adap/flower/tree/main/examples/app-pytorch). 


Running the example
-------------------

See `https://fedn.readthedocs.io/en/stable/quickstart.html` for a general introduction to FEDn. 
This example follows the same structure as the pytorch quickstart example. 

Build a virtual environment (note that you might need to install the 'venv' package): 

.. code-block::

   bin/init_venv.sh

Activate the virtual environment:

.. code-block::

   source .flower-client/bin/activate

Make the compute package (to be uploaded to FEDn):

.. code-block::

   tar -czvf package.tgz client

Create the seed model (to be uploaded to FEDn):
.. code-block::

   python client/entrypoint init_seed

Next, you will upload the compute package and seed model to
a FEDn network (a deployment of the server-side infrastructure). 

You have two main options: using FEDn Studio (SaaS)
(recommended for new users), or deploying a pseudo-local sandbox
on your own machine using docker compose. 

If you are using FEDn Studio (recommended):
-----------------------------------------------------

Follow instructions here to register for Studio and start a project: https://fedn.readthedocs.io/en/stable/studio.html.

In you Studio project: 

- From the "Sessions" menu, upload the compute package and seed model. 
- Register a client in Studio and obtain the corresponding 'client.yaml'.  

On your local machine / client (in the same virtual environment), start the FEDn client: 

.. code-block::

   CLIENT_NUMBER=0 FEDN_AUTH_SCHEME=Bearer fedn run client -in client.yaml --force-ssl --secure=True


Or, if you prefer to use Docker, build an image (this might take a long time):

.. code-block::

   docker build -t flower-client .

Then start the client usign Docker:

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   -e CLIENT_NUMBER=0 \
   -e FEDN_AUTH_SCHEME=Bearer \
   flower-client run client -in client.yaml --secure=True --force-ssl


If you are running FEDn in pseudo-local mode:
------------------------------------------------------------------

Deploy a FEDn network on local host (see `https://fedn.readthedocs.io/en/stable/quickstart.html`). 

Use the FEDn API Client to initalize FEDn with the compute package and seed model: 

.. code-block::

   python init_fedn.py

Create a file 'client.yaml' with the following content: 

.. code-block::
   
   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092
   name: myclient

Then start the client (using Docker)

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   --network=fedn_default \
   -e CLIENT_NUMBER=0 \
   flower-client run client -in client.yaml
