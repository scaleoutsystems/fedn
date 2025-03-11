ASYNC CLIENTS 
-------------

This example shows how to experiment with intermittent and asynchronous client workflows.     

Prerequisites
-------------

- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker (if running locally)](https://docs.docker.com/get-docker)
- [Docker Compose (if running locally)](https://docs.docker.com/compose/install)

Running the example
------------------

First, make sure that FEDn is installed (we recommend using a virtual environment)

Clone FEDn

.. code-block::

    git clone https://github.com/scaleoutsystems/fedn.git

Install FEDn

.. code-block::

    pip install fedn


Prepare the example environment and seed model
---------------------------------------------

Standing in the folder fedn/examples/async-clients

.. code-block::

    pip install -r requirements.txt

Create the seed model

.. code-block::

    python init_seed.py


You will now have a file 'seed.npz' in the directory.

Configuration
------------

The file ``config.py`` contains all configuration settings for this example. The most important setting is ``USE_LOCAL`` which determines whether to run with a local FEDn deployment or connect to a remote instance such as a Studio instance project.

For local deployment:
- Set ``USE_LOCAL = True`` in ``config.py``
- Deploy FEDn locally using Docker Compose (see below)

For remote deployment:
- Set ``USE_LOCAL = False`` in ``config.py``
- Set up a project in Scaleout Studio
- Update the ``DISCOVER_HOST`` in ``REMOTE_CONFIG`` with your API URL
- Create a ``tokens.json`` file with the following structure:

.. code-block::

    {
        "api.fedn.scaleoutsystems.com/your-project-name": {
            "CLIENT_TOKEN": "your-client-token-here",
            "ADMIN_TOKEN": "your-admin-token-here"
        }
    }

Replace ``your-project-name``, ``your-client-token-here``, and ``your-admin-token-here`` with your actual values.

Running a simulation
-------------------

For local deployment, start FEDn. Standing in the FEDn root directory:

.. code-block::

    docker-compose up 

Initialize FEDn with the seed model:

.. code-block::

    python init_fedn.py

Monitoring client status (optional)
----------------------------------

If you want to monitor client statuses, edit ``client_status.py`` and update the ``MACHINE_NAMES`` list with the names of the machines running your clients. Then run:

.. code-block::

    python client_status.py

This will periodically check and record client statuses to a CSV file.

Running clients and analyzing participation
------------------------------------------

Start simulating clients:

.. code-block::

    python run_clients.py

You can use the ``--intermittent`` flag to simulate clients that periodically disconnect and reconnect.

To analyze client participation and identify potential issues:

.. code-block::

    python client_participation.py

This will generate plots showing the number of aggregated models and validations per round, helping you understand client participation patterns and identify where things might be going wrong.

Start the experiment / training sessions: 

.. code-block::

    python run_experiment.py

Once global models start being produced, you can start analyzing results using API Client, refer to the notebook "Experiment.ipynb" for instructions.
