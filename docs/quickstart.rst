Getting started with FEDn
=========================

.. note::
   This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is designed to serve as a starting point for new developers. 
   To learn about FEDn Projects in order to develop your own federated machine learning projects, see :ref:`projects-label`. 
   
**Prerequisites**

-  `Python >=3.8, <=3.12 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__ 


Start a FEDn Studio Project
----------------------------

Start by creating an account in Studio. Head over to `fedn.scaleoutsystems.com/signup <https://fedn.scaleoutsystems.com/signup/>`_  and sign up.

Logged into Studio, do: 

1. Click on the "New Project" button in the top right corner of the screen.
2. Continue by clicking the "Create button". The FEDn template contains all the services necessary to start a federation.
3. Enter the project name (mandatory). The project description is optional.
4. Click the "Create" button to create the project.

.. image:: img/studio_project_overview.png

When these steps are complete, you will see a Studio project similar to the above image. The Studio project provides all server side components of FEDn needed to manage 
federated training. We will use this project in a later stage to run the federated experiments. But first, we will set up the local client.

Install FEDn on your client
----------------------------

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


Create the compute package and seed model
--------------------------------------------

Next, we will prepare the client. For illustrative purposes, we use one of the pre-defined projects in the FEDn repository, ``minst-pytorch``. 

In order to train a federated model using FEDn, your Studio project needs to be initialized with a ``compute package`` and a ``seed model``. The compute package is a code bundle containing the 
code used by the client to execute local training and local validation. The seed model is a first version of the global model.  

Locate into ``examples/mnist-pytorch`` folder in the cloned fedn repository. The compute package is located in the folder ``client``.

Create a package of the fedn project. Standing in ``examples/mnist-pytorch``: 

.. code-block::

   fedn package create --path client

This will create a package called ``package.tgz`` in the root of the project.

Next, create the seed model: 

.. code-block::

   fedn run build --path client

This will create a seed model called ``seed.npz`` in the root of the project. We will now upload these to your Studio project using the FEDn APIClient. 

For a detailed explaination of the FEDn Project with instructions for how to create your own project, see this guide: :ref:`projects-label`

Initialize your FEDn Studio Project
------------------------------------

In the Studio UI, navigate to the project you created above and click on the "Sessions" tab. Click on the "New Session" button. Under the "Compute package" tab, select a name and upload the generated package file. Under the "Seed model" tab, upload the generated seed file:

.. image:: img/upload_package.png

**Upload the package and seed model using the Python APIClient**

It is also possible to upload a package and seed model using the Python API Client. 

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

Each local client needs an access token in order to connect. These tokens are issued from your Studio Project. Go to the Clients' tab and click 'Connect client'.
Download a client configuration file and save it to the root of the ``examples/mnist-pytorch folder``. Rename the file to 'client.yaml'.
Then start the client by running the following command:

.. code-block::

   fedn run client -in client.yaml --secure=True --force-ssl

Repeat the above for the number of clients you want to use. A normal laptop should be able to handle several clients for this example.

**Modifying the data split (multiple-clients, optional):**

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

In Studio click on the "Sessions" link, then the "New session" button in the upper right corner. Click the "Start session" tab and enter your desirable settings (or use default) and hit the "Start run" button. In the terminal where your are running your client you should now see some activity. When the round is completed, you can see the results in the FEDn Studio UI on the "Models" page.

**Watch the training progress**

Once a training session is started, you can monitor the progress of the training by navigating to "Sessions" and click on the "Open" button of the active session. The session page will list the models as soon as they are generated. To get more information about a particular model, navigate to the model page by clicking the model name. From the model page you can download the model weights and get validation metrics.

To get an overview of how the models have evolved over time, navigate to the "Models" tab in the sidebar. Here you can see a list of all models generated across sessions along with a graph showing some metrics of how the models are performing.

.. image:: img/studio_model_overview.png

.. _studio-api:

**Control training sessions using the Python APIClient**

You can also issue training sessions using the APIClient:

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


Please see :py:mod:`fedn.network.api` for more details on how to use the APIClient. 

Access model updates  
--------------------

.. note::
   In FEDn Studio, you can access global model updates by going to the 'Models' or 'Sessions' tab. Here you can download model updates, metrics (as csv) and view the model trail.


You can also access global model updates via the APIClient:

.. code:: python

   >>> ...
   >>> client.download_model("<model-id>", path="model.npz")


Connecting clients using Docker
--------------------------------

You can also use Docker to containerize the client. 
For convenience, there is a Docker image hosted on ghrc.io with fedn preinstalled.
To start a client using Docker: 

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -e FEDN_PACKAGE_EXTRACT_DIR=package \
     -e FEDN_NUM_DATA_SPLITS=2 \
     -e FEDN_DATA_PATH=/app/package/data/clients/1/mnist.pt \
     ghcr.io/scaleoutsystems/fedn/fedn:0.10.0 run client -in client.yaml --force-ssl --secure=True


Where to go from here?
------------------------

With you first FEDn federated project set up, we suggest that you take a close look at how a FEDn project is structured
and how you develop your own FEDn projects:

- :ref:`projects-label`

You can also dive into the architecture overview to learn more about how FEDn is designed and works under the hood: 
- :ref:`architecture-label`


