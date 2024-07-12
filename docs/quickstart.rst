Getting started with FEDn
=========================

.. note::
   This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is designed to serve as a starting point for new developers. 
   To learn how to develop your own project from scratch, see :ref:`projects-label`. 
   
**Prerequisites**

-  `Python >=3.8, <=3.12 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__ 


1. Start a FEDn Studio Project
------------------------------

Start by creating an account in Studio. Head over to `fedn.scaleoutsystems.com/signup <https://fedn.scaleoutsystems.com/signup/>`_  and sign up.

Logged into Studio, create a new project by clicking  on the "New Project" button in the top right corner of the screen.
You will see a Studio project similar to the image below. The Studio project provides all the necessary server side components of FEDn. 
We will use this project in a later stage to run the federated experiments. But first, we will set up the local client.


.. image:: img/studio_project_overview.png


2. Install FEDn on your client
-------------------------------

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

Next, we will prepare the client. We will use one of the pre-defined projects in the FEDn repository, ``mnist-pytorch``. 

3. Create the compute package and seed model
--------------------------------------------

In order to train a federated model using FEDn, your Studio project needs to be initialized with a ``compute package`` and a ``seed model``. The compute package is a code bundle containing the 
code used by the client to execute local training and local validation. The seed model is a first version of the global model. 
For a detailed explaination of the compute package and seed model, see this guide: :ref:`projects-label`

To work through this quick start you need a local copy of the ``mnist-pytorch`` example project contained in the main FEDn Git repository. 
The following command clones the entire repository but you will only use the example:

.. code-block:: bash

   git clone https://github.com/scaleoutsystems/fedn.git

Locate into the ``fedn/examples/mnist-pytorch`` folder. The compute package is located in the folder ``client``.

Create a compute package: 

.. code-block::

   fedn package create --path client

This will create a file called ``package.tgz`` in the root of the project.

Next, create the seed model: 

.. code-block::

   fedn run build --path client

This will create a file called ``seed.npz`` in the root of the project. 

.. note::
   This example automatically creates the runtime environment for the compute package using Virtualenv. 
   When you first exectue the above commands, FEDn will build a venv, and this takes 
   a bit of time. For more information on the various options to manage the environement, see :ref:`projects-label`. 

Next will now upload these files to your Studio project:  

4. Initialize your FEDn Studio Project
--------------------------------------

In the Studio UI, navigate to the project you created above and click on the "Sessions" tab. Click on the "New Session" button. Under the "Compute package" tab, select a name and upload the generated package file. Under the "Seed model" tab, upload the generated seed file:

.. image:: img/upload_package.png

**Upload the package and seed model using the Python APIClient**

It is also possible to upload a package and seed model using the Python API Client. 

.. note:: 
   You need to create an API admin token and use the token to authenticate the APIClient.
   Do this by going to the 'Settings' tab in FEDn Studio and click 'Generate token'. Copy the access token and use it in the APIClient below.
   The controller host can be found on the main Dashboard in FEDn Studio. More information on the use of the APIClient can be found here: :ref:`apiclient-label.

To upload the package and seed model using the APIClient:

.. code:: python

   >>> from fedn import APIClient
   >>> client = APIClient(host="<controller-host>", token="<access-token>", secure=True, verify=True)
   >>> client.set_active_package("package.tgz", helper="numpyhelper")
   >>> client.set_active_model("seed.npz")


5. Configure and attach clients
-------------------------------

**Generate an access token for the client (in Studio)**

Each local client needs an access token in order to connect securely to the FEDn server. These tokens are issued from your Studio Project. 
Go to the Clients' tab and click 'Connect client'. Download a client configuration file and save it to the root of the ``examples/mnist-pytorch folder``. 
Rename the file to 'client.yaml'. 

**Start the client (on your local machine)** 

Now we can start the client by running the following command:

.. code-block::

   fedn run client -in client.yaml --secure=True --force-ssl

Repeat these two steps (generate an access token and start a local client) for the number of clients you want to use.
A normal laptop should be able to handle several clients for this example.

**Modifying the data split (multiple-clients, optional):**

The default traning and test data for this particular example (mnist-pytorch) is for convenience downloaded and split automatically by the client when it starts up (see the 'startup' entrypoint). 
The number of splits and which split to use by a client can be controlled via the environment variables ``FEDN_NUM_DATA_SPLITS`` and ``FEDN_DATA_PATH``.
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


6. Start a training session
---------------------------

In Studio click on the "Sessions" link, then the "New session" button in the upper right corner. Click the "Start session" tab and enter your desirable settings (the default settings are good for this example) and hit the "Start run" button.
In the terminal where your are running your client you should now see some activity. When a round is completed, you can see the results on the "Models" page.

**Watch the training progress**

Once a training session is started, you can monitor the progress of the training by navigating to "Sessions" and click on the "Open" button of the active session. The session page will list the models as soon as they are generated. 
To get more information about a particular model, navigate to the model page by clicking the model name. From the model page you can download the model weights and get validation metrics.

.. image:: img/studio_model_overview.png

.. _studio-api:

Congratulations, you have now completed your first federated training session with FEDn! Below you find additional information that can
be useful as you progress in your federated learning journey.

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

**Downloading global model updates**  

.. note::
   In FEDn Studio, you can access global model updates by going to the 'Models' or 'Sessions' tab. Here you can download model updates, metrics (as csv) and view the model trail.


You can also access global model updates via the APIClient:

.. code:: python

   >>> ...
   >>> client.download_model("<model-id>", path="model.npz")

**Where to go from here?**
--------------------------

With you first FEDn federated project set up, we suggest that you take a close look at how a FEDn project is structured
and how you develop your own FEDn projects:

- :ref:`projects-label`

You can also dive into the architecture overview to learn more about how FEDn is designed and works under the hood: 

- :ref:`architecture-label`

For developers looking to cutomize FEDn and develop own aggregators, check out the local development guide. 
This page also has instructions for using Docker to run clients: 

- :ref:`developer-label`





