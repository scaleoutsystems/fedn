.. _quickstart-label:

Getting started with Scaleout Edge
==================================

.. note::
   This quickstart guide will help you get started with the Scaleout Edge platform using an existing or pre-provisioned deployment.  
   If you don't yet have access to a project, follow the steps below to request one.  
   To learn how to develop and configure your own project from scratch, see :ref:`projects-label`. 

**Prerequisites**

-  `Python >=3.9, <=3.12 <https://www.python.org/downloads>`__

1. Get a project
-----------------

Before you can start using Scaleout Edge, you’ll need access to a project.  
Projects define the environment where your federated applications run and are hosted and managed by Scaleout Systems, unless you choose an on-prem deployment. You can request a new project through our online request form.

#. **Request a project.** Fill out the form at `scaleoutsystems.com/request-project <https://scaleoutsystems.com/request-project/>`_ to tell us about your use case and preferred deployment option.
#. **Choose your deployment.** We offer several hosting options:
   - **Academic (free)** — for research and educational collaborations, hosted by Scaleout Systems.
   - **Enterprise** — for organizations that require dedicated infrastructure. Enterprise projects can be **on-prem** (self-hosted) or **fully managed** by Scaleout Systems.
#. **Wait for confirmation.** Our team will review your request and contact you with details on setup and access.
#. **Start collaborating.** Once your project is approved and provisioned, you’ll receive credentials and connection details to begin integrating your clients and nodes.

1.5 Set up a Virtual environment (Recommended)
----------------------------------------------

Before installing Scaleout Edge using pip, we recommend creating a virtual environment. This helps isolate dependencies and avoids conflicts with other Python projects on your machine.

You can set up and activate a virtual environment using the following steps:

**Using venv** (Python's built-in module for virtual environments)

.. tabs::

    .. code-tab:: bash
         :caption: Unix/MacOS

         python3 -m venv scaleout_env
         source scaleout_env/bin/activate

    .. code-tab:: bash
         :caption: Windows (PowerShell)

         python -m venv scaleout_env
         Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
         scaleout_env\Scripts\Activate.ps1

    .. code-tab:: bash
         :caption: Windows (CMD.exe)

         python -m venv scaleout_env
         scaleout_env\Scripts\activate.bat


For additional information visit the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_. 

After activating the virtual environment, you can proceed with the next steps.

2. Prepare the clients and define the global model
---------------------------------------------------

Next, we will prepare and package the ML code to be executed by each client and create a first version of the global model (seed model). 
We will work with one of the pre-defined projects in the Scaleout client repository, ``mnist-pytorch``. 

First install the Scaleout Edge API on your local machine (client): 

**Using pip**

On you local machine/client, install the Scaleout Edge package using pip:

.. code-block:: bash

   pip install scaleout

**From source**

Clone the Scaleout Client repository and install the package:

.. code-block:: bash

   git clone https://github.com/scaleoutsystems/scaleout-client.git
   cd scaleout-client
   pip install .


.. _package-creation:

**Create the compute package and seed model**

In order to train a federated model using Scaleout Edge, your project needs to be initialized with a ``compute package`` and a ``seed model``. The compute package is a code bundle containing the 
code used by the client to execute local training and local validation. The seed model is a first version of the global model. 
For a detailed explaination of the compute package and seed model, see this guide: :ref:`projects-label`

To work through this quick start you need a local copy of the ``mnist-pytorch`` example project contained in the main Scaleout Edge Git repository. 
Clone the repository using the following command, if you didn't already do it in the previous step:

.. code-block:: bash

   git clone https://github.com/scaleoutsystems/scaleout-client.git

Navigate to the ``scaleout-client/python/examples/mnist-pytorch`` folder. The compute package is located in the folder ``client``.

Create a compute package: 

.. code-block::

   scaleout package create --path client

This will create a file called ``package.tgz`` in the root of the project.

Next, create the seed model: 

.. code-block::

   scaleout run build --path client

This will create a file called ``seed.npz`` in the root of the project. 

.. note::
   This example automatically creates the runtime environment for the compute package using Virtualenv. 
   When you first exectue the above commands, Scaleout Edge will build a venv, and this takes 
   a bit of time. For more information on the various options to manage the environement, see :ref:`projects-label`. 

Next will now upload these files to your Scaleout Edge project.  

3. Initialize the server-side
------------------------------
The next step is to initialize the server side with the client code and the initial global model. In the deployment UI,

**Upload the compute package**

#. Navigate to your project from Step 1 and click Packages in the sidebar.
#. Click Add Package.
#. In the form that appears, enter a name and upload the generated package file.

.. note:: 
   If no compute package is selected, the system will run in local mode. This is an advanced option that 
   allows each client to connect with their own custom training and validation logic. It can also be useful during development, as it eliminates the need to upload a new package with every change or version update.


**Upload the seed model**

#. Click on the Models tab in the sidebar.
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
The number of splits and which split to use by a client can be controlled via the environment variables ``SCALEOUT_NUM_DATA_SPLITS`` and ``SCALEOUT_DATA_PATH``.

Setup the environement for a client (using a 10-split and the 1st partition) by running the following commands:

.. tabs::

    .. code-tab:: bash
         :caption: Unix/MacOS

         export SCALEOUT_PACKAGE_EXTRACT_DIR=package
         export SCALEOUT_NUM_DATA_SPLITS=10
         export SCALEOUT_DATA_PATH=./data/clients/1/mnist.pt

    .. code-tab:: bash
         :caption: Windows (PowerShell)

         $env:SCALEOUT_PACKAGE_EXTRACT_DIR=".\package"
         $env:SCALEOUT_NUM_DATA_SPLITS=10
         $env:SCALEOUT_DATA_PATH=".\data\clients\1\mnist.pt"

    .. code-tab:: bash
         :caption: Windows (CMD.exe)

         set SCALEOUT_PACKAGE_EXTRACT_DIR=.\package\\
         set SCALEOUT_NUM_DATA_SPLITS=10
         set SCALEOUT_DATA_PATH=.\data\\clients\\1\\mnist.pt

**Start the client (on your local machine)** 

Each local client requires an access token to connect securely to the Scaleout Edge server. These tokens are issued from your Scaleout Edge Project. 

#. Navigate to the Clients page and click Connect Client.
#. Follow the instructions in the dialog to generate a new token.
#. Copy and paste the provided command into your terminal to start the client.

Repeat these two steps for the number of clients you want to use.
A normal laptop should be able to handle several clients for this example. Remember to use different partitions for each client, by changing the number in the ``SCALEOUT_DATA_PATH`` variable. 

5. Train the global model 
-----------------------------

With clients connected, we are now ready to train the global model.

.. tip::

   You can use the Scaleout Edge API Client to start a session and monitor the progress. For more details, see :ref:`apiclient-label`.

   .. code-block:: python

      client.start_session(name="My Session", rounds=5)


In the Scaleout Edge UI, 

#. Navigate to the Sessions page and click on "Create session". Fill in the form with the desired settings.
#. When the session is created, click "Start training" and select the number of rounds to run.
#. Once the training is started, you can follow the progress in the UI.

In the terminal where your are running your client you should now see some activity. When a round is completed, you can see the results on the "Models" page.

Congratulations, you have now completed your first federated training session with Scaleout Edge! Below you find additional information that can
be useful as you progress in your federated learning journey.

.. note::
   In the Scaleout Edge UI, you can access global model updates by going to the 'Models' or 'Sessions' tab. Here you can download model updates, metrics (as csv) and view the model trail.

**Where to go from here?**

With you first Scaleout Edge federated project set up, we suggest that you take a closer look at how a Scaleout Edge project is structured
to learn how to develop your own Scaleout Edge projects:

:ref:`projects-label`

In this tutorial we relied on the UI for running training sessions and retrieving models and results. 
The Python APIClient provides a flexible alternative, with additional functionality exposed, 
including the use of different aggregators. Learn how to use the APIClient here: 

:ref:`apiclient-label`

Study the architecture overview to learn more about how Scaleout Edge is designed and works under the hood: 

:ref:`architecture-label`

.. meta::
   :description lang=en: This tutorial is a quickstart guide to Scaleout Edge based on a pre-made Scaleout Edge Project. It is designed to serve as a starting point for new developers.
   :keywords: Getting started with Federated Learning, Federated Learning, Federated Learning Framework, Federated Learning Platform
   :og:title: Getting started with Scaleout Edge
   :og:description: This tutorial is a quickstart guide to Scaleout Edge based on a pre-made Scaleout Edge Project. It is designed to serve as a starting point for new developers.
   :og:url: https://docs.scaleoutsystems.com/en/stable/quickstart.html
   :og:type: website
