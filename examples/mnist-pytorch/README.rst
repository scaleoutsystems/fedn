Quickstart Tutorial PyTorch (MNIST)
-----------------------------------

This is an example FEDn Project based on the classic hand-written text recognition dataset MNIST.
The project is well suited as a lightweight test when developing on FEDn in pseudo-distributed mode. 
A normal high-end laptop or a workstation (CPU only or GPU) should be able to sustain a few clients. 
The example automates the partitioning of data and deployment of a variable number of clients on a single host. 
We here assume working experience with containers, Docker and docker-compose. 
   
Prerequisites
-------------

Using FEDn Studio:

-  `Python 3.8, 3.9, 3.10 or 3.11 <https://www.python.org/downloads>`__
-  A FEDn Studio account: https://fedn.scaleoutsystems.com/signup   

If using self-managed with docker-compose:

-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Quick start with FEDn Studio
----------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch

Create the compute package (compress the 'client' folder):

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in the global model trail):

.. code-block::

   fedn run build --path client

This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  

Follow the guide here to set up your FEDn Studio project: https://fedn.readthedocs.io/en/latest/studio.html. On the 
step "Upload Files", upload 'package.tgz' and 'seed.npz'. 

In Studio, go to "Clients" and download a new client configuration file (contains the access token). 
Then, start the client using the client.yaml file:

.. code-block::

   export FEDN_PACKAGE_EXTRACT_DIR=package
   fedn run client -in client.yaml --secure=True --force-ssl

The default traning and test data is for this example downloaded and split direcly by the client when it starts up. 
The data will be found in package/data/clients/1/mnist.pt and can be changed to other partitions by exporting the environment variable FEDN_DATA_PATH.
For example, to use the second partiton:

.. code-block::

   export FEDN_DATA_PATH=data/clients/2/mnist.pt

The default split into 2 partitions can be changed in client/data.py.

Connecting clients using Docker
===============================

For convenience, there is a Docker image hosted on ghrc.io with fedn preinstalled. To start a client using Docker: 

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -e FEDN_PACKAGE_EXTRACT_DIR=package \
     -e FEDN_NUM_DATA_SPLITS=2 \
     -e FEDN_DATA_PATH=/app/package/data/clients/1/mnist.pt \
     ghcr.io/scaleoutsystems/fedn/fedn:0.9.0 run client -in client.yaml --force-ssl --secure=True


Working in psuedo-distributed mode (for local development)
----------------------------------------------------------

Follow the steps above to install FEDn, generate 'package.tgz' and 'seed.tgz'.

Start a pseudo-distributed FEDn network using docker-compose:

.. code-block::

   docker compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    up

This starts up local services for MongoDB, Minio, the API Server, one Combiner and two clients. 
You can verify the deployment using these urls: 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081

Upload the package and seed model to FEDn controller using the APIClient:

.. code:: python

   >>> from fedn import APIClient
   >>> client = APIClient(host="localhost", port=8092)
   >>> client.set_active_package("package.tgz", helper="numpyhelper")
   >>> client.set_active_model("seed.npz")

The client should now recieve and unpack the compute package, and report "Client is active, waiting for model update requests".
You can now start a training session with 5 rounds (default): 

.. code:: python

   >>> client.start_session()

Automate experimentation with several clients  
=============================================

If you want to scale the number of clients, you can do so by modifying ``docker-compose.override.yaml``. For example, 
in order to run with 3 clients, change the envinronment variable ``FEDN_NUM_DATA_SPLITS`` to 3, and add one more client 
by copying ``client1`` and setting ``FEDN_DATA_PATH`` to ``/app/package/data/clients/3/mnist.pt``


Access message logs and validation data from MongoDB  
====================================================

You can access and download event logs and validation data via the API, and you can also as a developer obtain 
the MongoDB backend data using pymongo or via the MongoExpress interface: 

- http://localhost:8081/db/fedn-network/ 

The credentials are as set in docker-compose.yaml in the root of the repository. 

Access global models   
====================

You can obtain global model updates from the 'fedn-models' bucket in Minio: 

- http://localhost:9000

Reset the FEDn deployment   
=========================

To purge all data from a deployment incuding all session and round data, access the MongoExpress UI interface and 
delete the entire ``fedn-network`` collection. Then restart all services. 

Clean up
========
You can clean up by running 

.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml down -v
