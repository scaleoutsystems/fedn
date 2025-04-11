.. _fedn-cli:

CLI
=================================

The FEDN Command-Line Interface (CLI) is a powerful tool that allows users to interact with the FEDN platform. It provides a comprehensive set of commands to manage and operate various components of the FEDN network, including starting services, managing sessions, and retrieving data.

With the FEDN CLI, users can:

- Start and manage FEDN services such as the **combiner**, **controller**, and **clients**.
- Interact with the **controller** to:
  - Manage sessions, including starting, stopping, and monitoring their progress.
  - Retrieve data and results related to sessions, such as aggregated models and validation metrics.
  - Query the state of the network, including the status of connected combiners and clients.
- Test entry points in a FEDN package:
  - For example, use the CLI to test the script defined in the `train` entry point of a FEDN package. This allows users to validate and debug their training scripts in isolation before deploying them in a federated learning session.

The FEDN CLI is designed to streamline the management of the FEDN platform, making it easier for users to deploy, monitor, and interact with their federated learning networks.

For detailed usage and examples, refer to the sections below.

Client
------

The `fedn client` commands allow users to start and manage FEDN clients. Clients are the entities that participate in federated learning sessions and contribute their local data and models to the network.

**Commands:**

- **fedn client start** - Start a FEDN client using a specified configuration file or package. Example: 
   
.. code-block:: bash

    fedn client start --init client_config.yaml --local-package

- **fedn client list** - List all active FEDN clients in the network. Example:  

.. code-block:: bash
     
     fedn client list

- **fedn client get-config** - Get the configuration of a specific FEDN client from Studio, including the client's token and other details. Example:  

.. code-block:: bash
     
     fedn client get-config --name test-client

Combiner
--------

The `fedn combiner` commands allow users to start and manage combiners, which aggregate models from clients in the network.

**Commands:**

- **fedn combiner start** - Start a FEDN combiner using a specified configuration file. Example:  

.. code-block:: bash

     fedn combiner start --config combiner_config.yaml

Controller
----------

The `fedn controller` commands allow users to start and manage the FEDN controller, which orchestrates the entire federated learning process.

**Commands:**

- **fedn controller start** - Start the FEDN controller using a specified configuration file. Example:  

.. code-block:: bash

     fedn controller start --config controller_config.yaml

Studio
------

The `fedn studio` commands allow users to log in to the FEDN Studio and interact with the platform.

**Commands:**

- **fedn studio login** - Log in to the FEDN Studio using a username, password, and host. Example:  

.. code-block:: bash

     fedn studio login -u username -P password -H studio_host

Project
-------

The `fedn project` commands allow users to create, delete, list, and set the context for projects in the FEDN Studio.

**Commands:**

- **fedn project create** - Create a new project in the FEDN Studio. Example:  

.. code-block:: bash

     fedn project create -n project_name -H studio_host

- **fedn project delete** - Delete an existing project. Example:  

.. code-block:: bash

     fedn project delete -id project_id -H studio_host

- **fedn project list** - List all projects in the FEDN Studio. Example:  

.. code-block:: bash

     fedn project list -H studio_host

- **fedn project set-context** - Set the context for a specific project. Example:  

.. code-block:: bash

     fedn project set-context -id project_id -H studio_host

Model
-----

The `fedn model` commands allow users to manage models in the FEDN Studio.

**Commands:**

- **fedn model set-active** - Set a specific model as the active model for a project. Example:  

.. code-block:: bash

     fedn model set-active -f model_file.npz -H studio_host

- **fedn model list** - List all models in the FEDN Studio. Example:  

.. code-block:: bash

     fedn model list -H studio_host

Package
-------

The `fedn package` commands allow users to create and list packages in the FEDN Studio.

**Commands:**

- **fedn package create** - Create a new package for a project. Example:  

.. code-block:: bash

     fedn package create -n package_name -H studio_host

- **fedn package list** - List all packages in the FEDN Studio. Example:  

.. code-block:: bash

     fedn package list -H studio_host

Session
-------

The `fedn session` commands allow users to start and list sessions in the FEDN Studio.

**Commands:**

- **fedn session start** - Start a new session for a project. Example:  

.. code-block:: bash

     fedn session start -n session_name -H studio_host

- **fedn session list** - List all sessions in the FEDN Studio. Example:  

.. code-block:: bash

     fedn session list -H studio_host

Validation
----------

The `fedn validation` commands allow users to retrieve and list validation results.

**Commands:**

- **fedn validation get** - Retrieve validation results for a specific round. Example:  

.. code-block:: bash

     fedn validation get -r round_number -H studio_host

- **fedn validation list** - List all validation results for a project. Example:  

.. code-block:: bash

     fedn validation list -H studio_host
