FEDn Project: Flower ClientApps in FEDn
---------------------------------------

This example demonstrates how to run a Flower 'ClientApp' on FEDn. Sign up to FEDn Studio for a quick and easy way to set up all the backend services: https://scaleout.scaleoutsystems.com/signup/ (optional).

The FEDn compute package 'client/entrypoint' 
uses a built-in Flower compatibiltiy adapter for convenient wrapping of the Flower client.
See `flwr_client.py` and `flwr_task.py` for the Flower client code (which is adapted from 
https://github.com/adap/flower/tree/main/examples/app-pytorch).


Running the example
-------------------

See `https://scaleout.readthedocs.io/en/stable/quickstart.html` for a general introduction to FEDn. 
This example follows the same structure as the pytorch quickstart example. 

Install fedn:

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout.git
   cd fedn/examples/flower-client

Create the compute package (compress the 'client' folder):

.. code-block::

   scaleout package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in the global model trail):

.. code-block::

   scaleout run build --path client

This creates a seed.npz in the root of the project folder. Next, you will upload the compute package and seed model to
a FEDn network. Here you have two main options: using FEDn Studio 
(recommended for new users), or a self-managed pseudo-distributed deployment
on your own machine. 

Using FEDn Studio:
-------------------------------------------

Follow instructions here to register for Studio and start a project: https://scaleout.readthedocs.io/en/stable/quickstart.html.

In your Studio project: 

- From the "Sessions" menu, upload the compute package (package.tgz) and seed model (seed.npz). 
- Register a client and obtain the corresponding 'client.yaml'.  

On your local machine / client, start the FEDn client: 


.. code-block::

   scaleout client start -in client.yaml --secure=True --force-ssl


Or, if you prefer to use Docker (this might take a long time):

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   -e CLIENT_NUMBER=0 \
   -e SCALEOUT_PACKAGE_EXTRACT_DIR=package \
   ghcr.io/scaleoutsystems/fedn/fedn:0.11.1 run client -in client.yaml --secure=True --force-ssl

Scaling to multiple clients
------------------------------------------------------------------

To scale the experiment with additional clients on the same host, generate another 'client.yaml' and execute the run command
again from another terminal. Inject a client number as an environment 
varible which is used for distributing data (see 'flwr_task.py').

For Unix Operating Systems:

.. code-block::

   CLIENT_NUMBER=0 scaleout run client -in client.yaml --secure=True --force-ssl

Using Windows PowerShell:

.. code-block::

   & { $env:CLIENT_NUMBER="0"; scaleout run client -in client.yaml --secure=$true --force-ssl }
