FEDn Project: MNIST (PyTorch)
-----------------------------

This is an example FEDn Project based on the classic hand-written text recognition dataset MNIST.
The example is intented as a minimalistic quickstart and automates the handling of training data
by letting the client download and create its partition of the dataset as it starts up.

   **Note: These instructions are geared towards users seeking to learn how to work
   with FEDn in local development mode using Docker/docker-compose. We recommend all new users
   to start by following the Quickstart Tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html**

Prerequisites
-------------

Using FEDn Studio:

-  `Python 3.8, 3.9, 3.10 or 3.11 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__

If using pseudo-distributed mode with docker-compose:

-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Creating the compute package and seed model
-------------------------------------------

Install fedn:

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch

Create the compute package:

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in a global model trail):

.. code-block::

   fedn run build --path client

This will create a seed model called 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).

Using FEDn Studio
-----------------

Follow the guide here to set up your FEDn Studio project and learn how to connect clients (using token authentication): `Studio guide <https://fedn.readthedocs.io/en/stable/studio.html>`__.
On the step "Upload Files", upload 'package.tgz' and 'seed.npz' created above.


Modifing the data split:
========================

The default traning and test data  for this example is downloaded and split direcly by the client when it starts up (see 'startup' entrypoint).
The number of splits and which split used by a client can be controlled via the environment variables ``FEDN_NUM_DATA_SPLITS`` and ``FEDN_DATA_PATH``.
For example, to split the data in 10 parts and start a client using the 8th partiton:

.. code-block::

   export FEDN_PACKAGE_EXTRACT_DIR=package
   export FEDN_NUM_DATA_SPLITS=10
   export FEDN_DATA_PATH=./data/clients/8/mnist.pt
   fedn client start -in client.yaml --secure=True --force-ssl

The default is to split the data into 2 partitions and use the first partition.


Connecting clients using Docker:
================================

For convenience, there is a Docker image hosted on ghrc.io with fedn preinstalled. To start a client using Docker:

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -e FEDN_PACKAGE_EXTRACT_DIR=package \
     -e FEDN_NUM_DATA_SPLITS=2 \
     -e FEDN_DATA_PATH=/app/package/data/clients/1/mnist.pt \
     ghcr.io/scaleoutsystems/fedn/fedn:0.9.0 run client -in client.yaml --force-ssl --secure=True


Local development mode using Docker/docker compose
--------------------------------------------------

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

Upload the package and seed model to FEDn controller using the APIClient. In Python:

.. code-block::

   from fedn import APIClient
   client = APIClient(host="localhost", port=8092)
   client.set_active_package("package.tgz", helper="numpyhelper")
   client.set_active_model("seed.npz")

You can now start a training session with 5 rounds (default):

.. code-block::

   client.start_session()

Automate experimentation with several clients
=============================================

If you want to scale the number of clients, you can do so by modifying ``docker-compose.override.yaml``. For example,
in order to run with 3 clients, change the environment variable ``FEDN_NUM_DATA_SPLITS`` to 3, and add one more client
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
