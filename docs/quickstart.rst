Getting started with FEDn
=========================

.. note::
   This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is designed to serve as a minimalistic starting point for developers. 
   To learn about FEDn Projects in order to develop your own federated machine learning projects, see :ref:`projects-label`. 
   
**Prerequisites**

-  `Python >=3.8, <=3.12 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__ 


Set up a FEDn Studio Project
----------------------------

Start by creating an account in Studio. Head over to `fedn.scaleoutsystems.com/signup <https://fedn.scaleoutsystems.com/signup/>`_  and sign up.

Logged into Studio, do: 

1. Click on the "New Project" button in the top right corner of the screen.
2. Continue by clicking the "Create button". The FEDn template contains all the services necessary to start a federation.
3. Enter the project name (mandatory). The project description is optional.
4. Click the "Create" button to create the project.

You Studio project provides all server side components. Next, you will set up your local machine / client and create a FEDn project.

Install FEDn
------------

**Using pip**

On you local machine/client, install the FEDn package using pip:

.. code-block:: bash

   pip install fedn

**From source**

Clone the FEDn repository and install the package:

.. code-block:: bash

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn
   pip install .

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

**Modifying the data split:**

The default traning and test data for this example (MNIST) is for convenience downloaded and split by the client when it starts up (see 'startup' entrypoint). 
The number of splits and which split is used by a client can be controlled via the environment variables ``FEDN_NUM_DATA_SPLITS`` and ``FEDN_DATA_PATH``.
For example, to split the data in 10 parts and start a client using the 8th partiton:

.. tabs::

    .. code-tab:: bash
         :caption: Unix/MacOS

         export FEDN_PACKAGE_EXTRACT_DIR=package
         export FEDN_NUM_DATA_SPLITS=10
         export FEDN_DATA_PATH=./data/clients/8/mnist.pt
         fedn client start -in client.yaml --secure=True --force-ssl

    .. code-tab:: bash
         :caption: Windows (Powershell)

         $env:FEDN_PACKAGE_EXTRACT_DIR="package"
         $env:FEDN_NUM_DATA_SPLITS=10
         $env:FEDN_DATA_PATH="./data/clients/8/mnist.pt"
         fedn client start -in client.yaml --secure=True --force-ssl


Start a training session
------------------------

You are now ready to start training the model using the APIClient:

.. code:: python

   >>> ...
   >>> client.start_session(id="test-session", rounds=3)
   # Wait for training to complete, when controller is idle:
   >>> client.get_controller_status()
   # Show model trail:
   >>> models = client.get_model_trail()
   # Show performance of latest global model:
   >>> model_id = models[-1]['model']
   >>> validations = client.get_validations(model_id=model_id)


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
