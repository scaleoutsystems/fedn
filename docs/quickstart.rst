Getting started with FEDn
=========================

.. note::
   This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is desinged to serve as a minimalistic starting point for developers. 
   To learn how to develop your own federated machine learning projects with FEDn, see :ref:`projects-label`. 

This classic example of hand-written text recognition is well suited as a lightweight test when developing on FEDn in pseudo-distributed mode or in FEDn Studio. 
A normal high-end laptop or a workstation should be able to sustain a few clients. 
The example automates the partitioning of data and deployment of a variable number of clients on a single host. 
We here assume working experience with containers, Docker and docker-compose if you are running in pseudo-distributed mode.
For a details on FEDn Studio UI, see :ref:`studio`. 
   
**Prerequisites using FEDn Studio (recommended)**

-  `Python >=3.8, <=3.11 <https://www.python.org/downloads>`__

**Prerequisites for pseudo-distributed mode**

-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__


In pseudo-distributed mode
--------------------------

.. note::
   This is not required if you are using FEDn Studio!

In pseudo-distributed mode, you can start a FEDn network using docker-compose.
Clone this repository, locate into it and start a pseudo-distributed FEDn network using docker-compose:

.. code-block::

   docker-compose up 

This starts up the needed backend services MongoDB and Minio, the API Server and one Combiner. 
You can verify the deployment using these urls: 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081

.. warning:: 
   The FEDn network is configured to use a local Minio and MongoDB instances for storage. This is not suitable for production, but is fine for testing.

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

Prepare the package and seed model
----------------------------------

Next, we will prepare the client. A key concept in FEDn is the package - 
a code bundle that contains entrypoints for training and (optionally) validating a model update on the client. 

Locate into ``examples/mnist-pytorch`` and familiarize yourself with the project structure. The dependencies needed in the client environment are specified in 
in ``client/python_env.yaml``.    

Create a package of the fedn project (assumes your current working directory is in the root of the project /examples/mnist-pytorch):

.. code-block::

   fedn package create --path client

This will create a package called 'package.tgz' in the root of the project.

Next, run the build entrypoint defined in ``client/fedn.yaml`` to build the model artifact.

.. code-block::

   fedn run build --path client

This will create a seed model called 'seed.npz' in the root of the project.

**Upload the package and seed model**

.. note:: 
   If you are using FEDn Studio, you need to create an admin token and use the token to authenticate the APIClient.
   Do this by going to the 'Settings' tab in FEDn Studio and click 'Generate token'. Copy the access token and use it in the APIClient.
   The controller host can be found on the dashboard in FEDn Studio.

   You can also upload the file via the FEDn Studio UI. Please see :ref:`studio-upload-files` for more details.

Upload the package and seed model to FEDn controller using the APIClient:

.. code:: python

   >>> from fedn import APIClient
   >>> client = APIClient(host="localhost", port=8092)
   >>> client.set_active_package("package.tgz", helper="numpyhelper")
   >>> client.set_active_model("seed.npz")

.. note::
   If you are using FEDn Studio, you need to authenticate the APIClient by setting the access token:
   
   .. code:: python

      client = APIClient(host=<controller-host>, token=<access-token>, secure=True, verify=True)

Configure and attach clients
----------------------------

**Pseudo-distributed mode**

In pseudo-distributed mode, you can start a client using the provided docker compose template in the root of the project.
.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up --scale client=2


This will build a container image for the client, start two clients and connect them to local API server.

.. note::

  In FEDn Studio, you can configure and attach clients to the network. Go to the 'Clients' tab and click 'Connect client'.
  Download the client configuration file and save it to the root of the examples/mnist-pytorch folder. Rename the file to 'client.yaml'.
  Then start the client by running the following command in the root of the project:

  .. code-block::

    export FEDN_AUTH_SCHEME=Bearer 
    fedn client start -in client.yaml --secure=True --force-ssl

Start a training session
------------------------

.. note:: 

   In FEDn Studio, you can start a training session by going to the 'Sessions' tab and click 'Start session'.

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

There is also a Jupyter `Notebook <https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb>`_ version of this tutorial including examples of how to fetch and visualize model validations.

Access logs and validation data from MongoDB  
--------------------------------------------
You can access and download event logs and validation data via the API. If your are running in pseudo-distributed mode, you can access the MongoDB backend directly.
Either using pymongo (or other mongo clients) or via the MongoExpress interface: 

- http://localhost:8081/db/fedn-network/ 

The credentials are as set in docker-compose.yaml in the root of the repository. 

Access model updates  
--------------------


.. note::
   In FEDn Studio, you can access model updates by going to the 'Models' or 'Sessions' tab. Here you can download model updates, metrics (as csv) and view the model trail.


You can access model updates via the APIClient:

.. code:: python

   >>> ...
   >>> client.download_model("<model-id>", path="model.npz")

.. note::
   If running in pseudo-distributed mode, you can access model updates via the Minio interface.
   You can obtain model updates from the 'fedn-models' bucket: 

   - http://localhost:9000


**Clean up**
If you are running in pseudo-distributed mode, you can stop the network using docker-compose:

.. code-block::

   docker-compose down

**Where to go from here?**

With you first FEDn federation deployed, we suggest that you take a close look at how a FEDn project is structured
and how you develop your own FEDn projects:

- :ref:`projects-label`
