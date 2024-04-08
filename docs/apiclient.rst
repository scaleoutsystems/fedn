APIClient
===============

FEDn comes with an *APIClient* for interacting with the FEDn network. The APIClient is a Python3 library that can be used to interact with the FEDn network programmatically. 

Installation
------------

The APIClient is available as a Python package on PyPI, and can be installed using pip:

.. code-block:: bash
   
   $ pip install fedn

Initialize the APIClient
------------------------

To initialize the APIClient, you need to provide the hostname and port of the FEDn API server. The default port is 8092. The following code snippet shows how to initialize the APIClient:

.. code-block:: python
   
   from fedn import APIClient
   client = APIClient("localhost", 8092)

Set active package and seed model
---------------------------------

The active package can be set using the following code snippet:

.. code-block:: python
   
   client.set_active_package(path="path/to/package.tgz", helper="numpyhelper")

To set the initial seed model, you can use the following code snippet:

.. code-block:: python
   
   client.set_active_model(path="path/to/seed.npz")

Start training session
----------------------

Once the active package and seed model are set, you can connect clients to the network and start training models. The following code snippet initializes a session (training rounds):

.. code-block:: python
   
   session = client.start_session(id="session_name")

List data
---------

Other than starting training sessions, the APIClient can be used to get data from the network, such as sessions, models etc. All entities are represented and they all work in a similar fashion.

* get_*() - (plural) list all entities of a specific type
* get_*(id=<id-of-entity>) - get a specific entity

Entities represented in the APIClient are:

* clients
* combiners
* models
* packages
* rounds
* sessions
* statuses
* validations

The following code snippet shows how to list all sessions:

.. code-block:: python
   
   sessions = client.get_sessions()

And the following code snippet shows how to get a specific session:

.. code-block:: python
   
   session = client.get_session(id="session_name")


For more information on how to use the APIClient, see the :py:mod:`fedn.network.api.client`, and the example `Notebooks <https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb>`_. 
