Getting started with FEDn
=========================

.. note::
   This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is designed to serve as a minimalistic starting point for developers. 
   To learn about FEDn Projects in order to develop your own federated machine learning projects, see :ref:`projects-label`. 
   
**Prerequisites**

-  `Python >=3.8, <=3.11 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://studio.scaleoutsystems.com/signup>`__ 


Set up a FEDn Studio Project
----------------------------

Start by creating an account in FEDn Studio and set up a project by following the instruction here: :ref:`studio`.

Install FEDn
------------

**Using pip**

Install the FEDn package using pip:

.. code-block:: bash

   pip install fedn

**From source**

Clone the FEDn repository and install the package:

.. code-block:: bash

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/fedn
   pip install -e .

It is recommended to use a virtual environment when installing FEDn.

.. _package-creation:

Initialize FEDn with the client code bundle and seed model 
----------------------------------------------------------

Next, we will prepare the client. The key part of a FEDn Project is the client definition - 
code that contains entrypoints for training and (optionally) validating a model update on the client. 

Locate into ``examples/mnist-pytorch`` and familiarize yourself with the project structure. The dependencies needed in the client environment are specified 
in ``client/python_env.yaml``. 

In order to train a federated model using FEDn, your Studio project needs to be initialized with a compute package and a seed model. The compute package is a bundle
of the client specification, and the seed model is a first version of the global model.  

Create a package of the fedn project (assumes your current working directory is in the root of the project /examples/mnist-pytorch):

.. code-block::

   fedn package create --path client

This will create a package called 'package.tgz' in the root of the project.

Next, run the build entrypoint defined in ``client/fedn.yaml`` to build the model artifact.

.. code-block::

   fedn run build --path client

This will create a seed model called 'seed.npz' in the root of the project. We will now upload these to your Studio project using the FEDn APIClient. 

**Upload the package and seed model**

.. note:: 
   You need to create an API admin token and use the token to authenticate the APIClient.
   Do this by going to the 'Settings' tab in FEDn Studio and click 'Generate token'. Copy the access token and use it in the APIClient below.
   The controller host can be found on the main Dashboard in FEDn Studio.

   You can also upload the file via the FEDn Studio UI. Please see :ref:`studio-upload-files` for more details.

Upload the package and seed model using the APIClient:

.. code:: python

   >>> from fedn import APIClient
   >>> client = APIClient(host="<controller-host>", token="<access-token>", secure=True, verify=True)
   >>> client.set_active_package("package.tgz", helper="numpyhelper")
   >>> client.set_active_model("seed.npz")


Configure and attach clients
----------------------------

Each local client needs an access token in order to connect. These tokens are issued from your Studio Project. Go to the 'Clients' tab and click 'Connect client'.
Download a client configuration file and save it to the root of the examples/mnist-pytorch folder. Rename the file to 'client.yaml'.
Then start the client by running the following command in the root of the project:

.. code-block::

   fedn run client -in client.yaml --secure=True --force-ssl

Repeat the above for the number of clients you want to use. A normal laptop should be able to handle several clients for this example.

**Modifing the data split:**

The default traning and test data for this example (MNIST) is for convenience downloaded and split by the client when it starts up (see 'startup' entrypoint). 
The number of splits and which split used by a client can be controlled via the environment variables ``FEDN_NUM_DATA_SPLITS`` and ``FEDN_DATA_PATH``.
For example, to split the data in 10 parts and start a client using the 8th partiton:

.. code-block::

   export FEDN_PACKAGE_EXTRACT_DIR=package
   export FEDN_NUM_DATA_SPLITS=10
   export FEDN_DATA_PATH=package/data/clients/8/mnist.pt
   fedn run client -in client.yaml --secure=True --force-ssl

Start a training session
------------------------

You are now ready to start training the model using the APIClient:

.. code:: python

   >>> ...
   >>> client.start_session(session_id="test-session", rounds=3)
   # Wait for training to complete, when controller is idle:
   >>> client.get_controller_status()
   # Show model trail:
   >>> client.get_model_trail()
   # Show model performance:
   >>> client.get_validations()


Please see :py:mod:`fedn.network.api` for more details on the APIClient. 

.. note:: 

   In FEDn Studio, you can start a training session by going to the 'Sessions' tab and click 'Start session'. See :ref:`studio` for a
   step-by-step guide for how to control experiments using the UI. 

Access model updates  
--------------------

.. note::
   In FEDn Studio, you can access global model updates by going to the 'Models' or 'Sessions' tab. Here you can download model updates, metrics (as csv) and view the model trail.


You can access global model updates via the APIClient:

.. code:: python

   >>> ...
   >>> client.download_model("<model-id>", path="model.npz")


**Connecting clients using Docker**

You can also use Docker to containerize the client. 
For convenience, there is a Docker image hosted on ghrc.io with fedn preinstalled.
To start a client using Docker: 

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -e FEDN_PACKAGE_EXTRACT_DIR=package \
     -e FEDN_NUM_DATA_SPLITS=2 \
     -e FEDN_DATA_PATH=/app/package/data/clients/1/mnist.pt \
     ghcr.io/scaleoutsystems/fedn/fedn:0.9.0 run client -in client.yaml --force-ssl --secure=True


**Where to go from here?**

With you first FEDn federation set up, we suggest that you take a close look at how a FEDn project is structured
and how you develop your own FEDn projects:

- :ref:`projects-label`


Local development deployment (using docker compose)
----------------------------------------------------------

.. note::
   These instructions are for users wanting to set up a local development deployment of FEDn (wihout Studio).
   This requires basic knowledge of Docker and docker-compose. 
   The main use-case for this is rapid iteration while developing the FEDn Project, 
   development of aggregator plugins, etc. 

Follow the steps above to install FEDn, generate 'package.tgz' and 'seed.tgz'. Then, instead of 
using a Studio project for a managed FEDn server-side, start a local FEDn network
using docker-compose:

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

**Automate experimentation with several clients**  

If you want to scale the number of clients, you can do so by modifying ``docker-compose.override.yaml``. For example, 
in order to run with 3 clients, change the environment variable ``FEDN_NUM_DATA_SPLITS`` to 3, and add one more client 
by copying ``client1`` and setting ``FEDN_DATA_PATH`` to ``/app/package/data/clients/3/mnist.pt``


**Access message logs and validation data from MongoDB**  

You can access and download event logs and validation data via the API, and you can also as a developer obtain 
the MongoDB backend data using pymongo or via the MongoExpress interface: 

- http://localhost:8081/db/fedn-network/ 

The credentials are as set in docker-compose.yaml in the root of the repository. 

**Access global models**  

You can obtain global model updates from the 'fedn-models' bucket in Minio: 

- http://localhost:9000

**Reset the FEDn deployment**   

To purge all data from a deployment incuding all session and round data, access the MongoExpress UI interface and 
delete the entire ``fedn-network`` collection. Then restart all services. 

**Clean up**

You can clean up by running 

.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml down -v
