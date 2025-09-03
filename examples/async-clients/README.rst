ASYNC CLIENTS 
-------------

This example shows how to experiment with intermittent and asynchronous client workflows.     

Prerequisites
-------------

- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker (if running locally)](https://docs.docker.com/get-docker)
- [Docker Compose (if running locally)](https://docs.docker.com/compose/install)

Set up environment
------------------

First, make sure that FEDn is installed (we recommend using a virtual environment)

Clone FEDn

.. code-block::

    git clone https://github.com/scaleoutsystems/fedn.git

Switch to the current branch
.. code-block::

    cd fedn
    git checkout feature/SK-1689

Install FEDn, standing in the root folder of FEDn

.. code-block::

    pip install -e .

Standing in the folder fedn/examples/async-clients

.. code-block::

    pip install -r requirements.txt

Upload seed model
------------

Create the seed model

.. code-block::

    python init_seed.py


You will now have a file 'seed.npz' in the directory. Add this seed model to the FEDn instance:

.. code-block::

    python init_fedn.py

Project configuration
------------

The file ``config.py`` contains all configuration settings for this example. The most important setting is ``USE_LOCAL`` which determines whether to run with a local FEDn deployment or connect to a remote instance such as a Studio instance project.

**For local deployment:**
- Set ``USE_LOCAL = True`` in ``config.py``
- Deploy FEDn locally using Docker Compose. Standing in the FEDn root directory:

.. code-block::

    docker compose up 

**For remote deployment:**
- Set ``USE_LOCAL = False`` in ``config.py``
- Set up a project in Scaleout Studio
- Update the ``DISCOVER_HOST`` in ``REMOTE_CONFIG`` with your API URL
- Update the ``ADMIN_TOKEN`` and ``CLIENT_TOKEN`` in ``REMOTE_CONFIG`` with your tokens accessible from the Studio project page

**For the 10k client reference deployment:**
- Set ``USE_LOCAL = False`` in ``config.py``
- Set ``USE_REFERENCE = True`` in ``config.py``
- Create a Studio deployment of the current github branch (feature/SK-1689) using the studio branch (feature/SK-1667). 
- Update the max_concurrent_clients in the configmap of the studio deployment to 10k+
- Attach a Nodeport service to the combiners.
- Set ``NodeIP = <your-node-ip>`` and ``DISCOVER_PORT = <your-NodePort>`` in ``REFERENCE_CONFIG`` in ``config.py``
- Update the ``DISCOVER_HOST`` in ``REMOTE_CONFIG`` with your API URL accessible from the Studio project page
- Update the ``ADMIN_TOKEN`` and ``CLIENT_TOKEN`` in ``REMOTE_CONFIG`` with your tokens accessible from the Studio project page


Running clients and analyzing participation
------------------------------------------

Start simulating clients:

.. code-block::

    python run_clients.py

In the config file you can set the number of clients to simulate with the ``N_CLIENTS`` parameter. You can also adjust the client behavior by modifying parameters such as ``CLIENTS_MAX_DELAY`` and ``CLIENTS_ONLINE_FOR_SECONDS`` to simulate different levels of client availability and reliability.

Start the experiment / training sessions: 

.. code-block::

    python run_experiment.py

You can adjust the number of sequential training sessions by modifying the ``N_SESSIONS`` parameter in ``config.py``. If you are using Scaleout Studio, you can also start a session directly through the Studio interface or use the APIClient.

Once global models start being produced, you can start analyzing results using API Client, refer to the notebook "Experiment.ipynb" for instructions or simpy use the Studio interface to visualize results.
