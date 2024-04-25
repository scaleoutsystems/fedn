Using Flower ClientApps in FEDn
-------------------------------

This example demonstrates how to run a Flower 'ClientApp' on FEDn.

The FEDn compute package 'client/entrypoint' 
uses a built-in Flower compatibiltiy adapter for convenient wrapping of the Flower client.
See `flwr_client.py` and `flwr_task.py` for the Flower client code (which is adapted from 
https://github.com/adap/flower/tree/main/examples/app-pytorch). 


Running the example
-------------------

See `https://fedn.readthedocs.io/en/stable/quickstart.html` for a general introduction to FEDn. 
This example follows the same structure as the pytorch quickstart example. 

Install fedn:

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/flower-client

Create the compute package (compress the 'client' folder):

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in the global model trail):

.. code-block::

   fedn run build --path client

Next, you will upload the compute package and seed model to
a FEDn network. Here you have two main options: using FEDn Studio 
(recommended for new users), or a pseudo-local deployment
on your own machine. 

If you are using FEDn Studio (recommended):
-----------------------------------------------------

Follow instructions here to register for Studio and start a project: https://fedn.readthedocs.io/en/stable/studio.html.

In your Studio project: 

- From the "Sessions" menu, upload the compute package and seed model. 
- Register a client and obtain the corresponding 'client.yaml'.  

On your local machine / client, start the FEDn client: 


.. code-block::

   export FEDN_AUTH_SCHEME=Bearer
   export FEDN_PACKAGE_EXTRACT_DIR=package
   CLIENT_NUMBER=0 fedn client start -in client.yaml --secure=True --force-ssl


Or, if you prefer to use Docker (this might take a long time):

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   -e CLIENT_NUMBER=0 \
   -e FEDN_AUTH_SCHEME=Bearer \
   -e FEDN_PACKAGE_EXTRACT_DIR=package \
   ghcr.io/scaleoutsystems/fedn/fedn:master run client -in client.yaml --secure=True --force-ssl


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
   -e FEDN_AUTH_SCHEME=Bearer \
   -e FEDN_PACKAGE_EXTRACT_DIR=package \
   ghcr.io/scaleoutsystems/fedn/fedn:master run client -in client.yaml
