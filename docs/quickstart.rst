.. _quickstart-label:

Getting started with FEDn
=========================

.. note::
   This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is designed to serve as a starting point for new developers. 
   To learn how to develop your own project from scratch, see :ref:`projects-label`. 
   
**Prerequisites**

-  `Python >=3.9, <=3.12 <https://www.python.org/downloads>`__

1. Set up project
-----------------

#. Create a FEDn account. Sign up at `fedn.scaleoutsystems.com/signup <https://fedn.scaleoutsystems.com/signup/>`_.
#. Verify your email. Check your inbox for a verification email and click the link to activate your account.
#. Log in and create a project. Once your account is activated, log in to the Studio and create a new project.
#. Manage your projects. If you have multiple projects, you can view and manage them here:  `fedn.scaleoutsystems.com/projects <https://fedn.scaleoutsystems.com/projects/>`_.

.. tip::

   You can also create a project using our CLI tool. Run the following command:
   For more details, see :doc:`cli`.

   .. code-block:: bash

      fedn project create --name "My Project"

   Replace `"My Project"` with your desired project name.


1.5 Set up a Virtual environment (Recommended)
----------------------------------------------

Before installing FEDn using pip, we recommend creating a virtual environment. This helps isolate dependencies and avoids conflicts with other Python projects on your machine.

You can set up and activate a virtual environment using the following steps:

**Using venv** (Python's built-in module for virtual environments)

.. tabs::

    .. code-tab:: bash
         :caption: Unix/MacOS

         python3 -m venv fedn_env
         source fedn_env/bin/activate

    .. code-tab:: bash
         :caption: Windows (PowerShell)

         python -m venv fedn_env
         Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
         fedn_env\Scripts\Activate.ps1

    .. code-tab:: bash
         :caption: Windows (CMD.exe)

         python -m venv fedn_env
         fedn_env\Scripts\activate.bat


For additional information visit the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_. 

After activating the virtual environment, you can proceed with the next steps.

2. Prepare the clients and define the global model
---------------------------------------------------

Next, we will prepare and package the ML code to be executed by each client and create a first version of the global model (seed model). 
We will work with one of the pre-defined projects in the FEDn repository, ``mnist-pytorch``. 

First install the FEDn API on your local machine (client): 

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


.. _package-creation:

**Create the compute package and seed model**

In order to train a federated model using FEDn, your Studio project needs to be initialized with a ``compute package`` and a ``seed model``. The compute package is a code bundle containing the 
code used by the client to execute local training and local validation. The seed model is a first version of the global model. 
For a detailed explaination of the compute package and seed model, see this guide: :ref:`projects-label`

To work through this quick start you need a local copy of the ``mnist-pytorch`` example project contained in the main FEDn Git repository. 
Clone the repository using the following command, if you didn't already do it in the previous step:

.. code-block:: bash

   git clone https://github.com/scaleoutsystems/fedn.git

Navigate to the ``fedn/examples/mnist-pytorch`` folder. The compute package is located in the folder ``client``.

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

Next will now upload these files to your Studio project.  

3. Initialize the server-side
------------------------------
The next step is to initialize the server side with the client code and the initial global model. In the Studio UI,

**Upload the compute package**

#. Navigate to your project from Step 1 and click Packages in the sidebar.
#. Click Add Package.
#. In the form that appears, enter a name and upload the generated package file.

.. note:: 
   If no compute package is selected, the system will run in local mode. This is an advanced option that 
   allows each client to connect with their own custom training and validation logic. It can also be useful during development, as it eliminates the need to upload a new package with every change or version update.


**Upload the seed model**

#. Navigate to your project from Step 1 and click Models in the sidebar.
#. Click Add Model.
#. In the form that appears, upload the generated seed model file.

.. note::
   You can upload multiple compute packages and seed models, selecting the appropriate one for each session. To create a new session from any model, navigate to its model page.

Continue to step 4 before starting the session. The uploaded package and seed files are saved.

4. Start clients
-----------------

Before starting the clients, we need to configure what data partition the clients should use. This way each client will have access to a unique subset of the data.

**Manage Data Splits for MNIST-PyTorch** 

The default training and test data for this particular example (mnist-pytorch) is for convenience downloaded and split automatically by the client when it starts up. 
The number of splits and which split to use by a client can be controlled via the environment variables ``FEDN_NUM_DATA_SPLITS`` and ``FEDN_DATA_PATH``.

Setup the environement for a client (using a 10-split and the 1st partition) by running the following commands:

.. tabs::

    .. code-tab:: bash
         :caption: Unix/MacOS

         export FEDN_PACKAGE_EXTRACT_DIR=package
         export FEDN_NUM_DATA_SPLITS=10
         export FEDN_DATA_PATH=./data/clients/1/mnist.pt

    .. code-tab:: bash
         :caption: Windows (PowerShell)

         $env:FEDN_PACKAGE_EXTRACT_DIR=".\package"
         $env:FEDN_NUM_DATA_SPLITS=10
         $env:FEDN_DATA_PATH=".\data\clients\1\mnist.pt"

    .. code-tab:: bash
         :caption: Windows (CMD.exe)

         set FEDN_PACKAGE_EXTRACT_DIR=.\package\\
         set FEDN_NUM_DATA_SPLITS=10
         set FEDN_DATA_PATH=.\data\\clients\\1\\mnist.pt

**Start the client (on your local machine)** 

Each local client requires an access token to connect securely to the FEDn server. These tokens are issued from your FEDn Project. 

#. Navigate to the Clients page and click Connect Client.
#. Follow the instructions in the dialog to generate a new token.
#. Copy and paste the provided command into your terminal to start the client.

Repeat these two steps for the number of clients you want to use.
A normal laptop should be able to handle several clients for this example. Remember to use different partitions for each client, by changing the number in the ``FEDN_DATA_PATH`` variable. 

5. Train the global model 
-----------------------------

With clients connected, we are now ready to train the global model.

.. tip::

   You can use the FEDn API Client to start a session and monitor the progress. For more details, see :ref:`apiclient-label`.

   .. code-block:: python

      client.start_session(name="My Session", rounds=5)


In the FEDn UI, 

#. Navigate to the Sessions page and click on "Create session". Fill in the form with the desired settings.
#. When the session is created, click "Start training" and select the number of rounds to run.
#. Once the training is started, you can follow the progress in the UI.

In the terminal where your are running your client you should now see some activity. When a round is completed, you can see the results on the "Models" page.

.. _studio-api:

Congratulations, you have now completed your first federated training session with FEDn! Below you find additional information that can
be useful as you progress in your federated learning journey.

.. note::
   In FEDn Studio, you can access global model updates by going to the 'Models' or 'Sessions' tab. Here you can download model updates, metrics (as csv) and view the model trail.

**Where to go from here?**

With you first FEDn federated project set up, we suggest that you take a closer look at how a FEDn project is structured
to learn how to develop your own FEDn projects:

:ref:`projects-label`

In this tutorial we relied on the UI for running training sessions and retrieving models and results. 
The Python APIClient provides a flexible alternative, with additional functionality exposed, 
including the use of different aggregators. Learn how to use the APIClient here: 

:ref:`apiclient-label`

Study the architecture overview to learn more about how FEDn is designed and works under the hood: 

:ref:`architecture-label`

For developers looking to customize FEDn and develop own aggregators, check out the local development guide
to learn how to set up an all-in-one development environment using Docker and docker-compose:

:ref:`developer-label`

.. meta::
   :description lang=en: This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is designed to serve as a starting point for new developers.
   :keywords: Getting started with Federated Learning, Federated Learning, Federated Learning Framework, Federated Learning Platform
   :og:title: Getting started with FEDn
   :og:description: This tutorial is a quickstart guide to FEDn based on a pre-made FEDn Project. It is designed to serve as a starting point for new developers.
   :og:image: https://fedn.scaleoutsystems.com/static/images/scaleout_black.png
   :og:url: https://fedn.scaleoutsystems.com/docs/quickstart.html
   :og:type: website
