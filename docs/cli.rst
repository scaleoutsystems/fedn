.. _fedn-cli:

CLI
=================================

The Scaleout Edge Command-Line Interface (CLI) is a powerful tool that allows users to interact with the Scaleout Edge server. It provides a comprehensive set of commands to manage and operate various components of the Scaleout Edge network, including starting services, managing sessions, and retrieving data.

With the Scaleout Edge CLI, users can:

- Start and manage Scaleout Edge services such as the **combiner**, **controller**, and **clients**.
- Interact with the **controller** to:
  - Manage sessions, including starting, stopping, and monitoring their progress.
  - Retrieve data and results related to sessions, such as aggregated models and validation metrics.
  - Query the state of the network, including the status of connected combiners and clients.
- Test entry points in a Scaleout Edge package:
  - For example, use the CLI to test the script defined in the `train` entry point of a Scaleout Edge package. This allows users to validate and debug their training scripts in isolation before deploying them in a federated learning session.

The Scaleout Edge CLI is designed to streamline the management of the Scaleout Edge platform, making it easier for users to deploy, monitor, and interact with their federated learning networks.

For detailed usage and examples, refer to the sections below.

Client
------

The `scaleout client` commands allow users to start and manage Scaleout Edge clients. Clients are the entities that participate in federated learning sessions and contribute their local data and models to the network.

**Commands:**

- **scaleout client start** - Start a Scaleout Edge client using a specified configuration file or package. Example: 
   
.. code-block:: bash

    scaleout client start --init client_config.yaml --local-package

- **scaleout client list** - List all active Scaleout Edge clients in the network. Example:  

.. code-block:: bash
     
     scaleout client list

- **scaleout client get-config** - Get the configuration of a specific client from Scaleout Edge, including the client's token and other details. Example:  

.. code-block:: bash
     
     scaleout client get-config --name test-client

Login
------

The `scaleout` commands allow users to log in to Scaleout Edge and interact with the platform.

**Commands:**

- **scaleout login** - Log in to the Scaleout Edge using a username, password, and host. Example:  

.. code-block:: bash

     scaleout login -u username -P password -H host

Combiner
--------

The `scaleout combiner` commands allow users to start and manage combiners, which aggregate models from clients in the network.

**Commands:**

- **scaleout combiner start** - Start a Scaleout Edge combiner using a specified configuration file. Example:  

.. code-block:: bash

     scaleout combiner start --config combiner_config.yaml

Controller
----------

The `scaleout controller` commands allow users to start and manage the Scaleout Edge controller, which orchestrates the entire federated learning process.

**Commands:**

- **scaleout controller start** - Start the Scaleout Edge controller using a specified configuration file. Example:  

.. code-block:: bash

     scaleout controller start --config controller_config.yaml

Model
-----

The `scaleout model` commands allow users to manage models in your Scaleout Edge project.

**Commands:**

- **scaleout model set-active** - Set a specific model as the active model for a project. Example:  

.. code-block:: bash

     scaleout model set-active -f model_file.npz -H host

- **scaleout model list** - List all models in your Scaleout Edge project. Example:  

.. code-block:: bash

     scaleout model list -H host

Package
-------

The `scaleout package` commands allow users to create and list packages in Scaleout Edge.

**Commands:**

- **scaleout package create** - Create a new package. Example:  

.. code-block:: bash

     scaleout package create -n package_name -H host

- **scaleout package list** - List all packages in your Scaleout Edge project. Example:  

.. code-block:: bash

     scaleout package list -H host

Session
-------

The `scaleout session` commands allow users to start and list sessions in Scaleout Edge.

**Commands:**

- **scaleout session start** - Start a new session for a project. Example:  

.. code-block:: bash

     scaleout session start -n session_name -H host

- **scaleout session list** - List all sessions in your Scaleout Edge project. Example:  

.. code-block:: bash

     scaleout session list -H host

Validation
----------

The `scaleout validation` commands allow users to retrieve and list validation results.

**Commands:**

- **scaleout validation get** - Retrieve validation results for a specific round. Example:  

.. code-block:: bash

     scaleout validation get -r round_number -H host

- **scaleout validation list** - List all validation results for a project. Example:  

.. code-block:: bash

     scaleout validation list -H host
