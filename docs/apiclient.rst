APIClient
===============

FEDn comes with an *APIClient* for interacting with the FEDn network. The APIClient is a Python3 library that can be used to interact with the FEDn network programmatically. 


The APIClient is available as a Python package on PyPI, and can be installed using pip:

.. code-block:: bash
   
   $ pip install fedn


To initialize the APIClient, you need to provide the hostname and port of the FEDn API server. The default port is 8092. The following code snippet shows how to initialize the APIClient:

.. code-block:: python
   
   from fedn import APIClient
   client = APIClient("localhost", 8092)

Before you can start training models, you need to set the active package and an initail seed model. The active package can be set using the following code snippet:

.. code-block:: python
   
   client.set_active_package(path="path/to/compute_package.zip", helper="numpyhelper")

To set the initial seed model, you can use the following code snippet:

.. code-block:: python
   
   client.set_active_model(path="path/to/seed_model.zip")

Once the active package and seed model are set, you can connect clients to the network and start training models. The following code snippet initializes a session (training rounds):

.. code-block:: python
   
   session = client.start_session(id="session_name")


For more information on how to use the APIClient, see the :py:mod:`fedn.network.api.client`, and the example `Notebooks <https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb>`_. 
