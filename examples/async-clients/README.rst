ASYNC CLIENTS 
-------------

This example shows how to experiment with intermittent and asynchronous client workflows.     

Prerequisites
-------------

- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

Running the example (pseudo-distributed, single host)
-----------------------------------------------------


First, make sure that FEDn is installed (we recommend using a virtual environment)

Clone FEDn

.. code-block::

    git clone https://github.com/scaleoutsystems/fedn.git

Install FEDn

.. code-block::

    pip install fedn


Prepare the example environment and seed model
-------------------------------------------------------------------

Standing in the folder fedn/examples/async-clients

.. code-block::

    pip install -r requirements.txt

Create the seed model

.. code-block::

    python init_seed.py seed.npz


You will now have a file 'seed.npz' in the directory.

Running a simulation
--------------------

Deploy FEDn on localhost. Standing in the the FEDn root directory

.. code-block::

    docker-compose up 


Initialize FEDn with the seed model

.. code-block::

    python init_fedn.py

Start simulating clients

.. code-block::

    python run_clients.py

Start the experiment / training sessions: 

.. code-block::

    python run_experiment.py

Once global models start being produced, you can start analyzing results using API Client, refer to the notebook "Experiment.ipynb" for instructions. 



