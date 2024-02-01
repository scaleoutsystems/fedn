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

For more information on how to use the APIClient, see the :py:mod:`fedn.network.api.client`, and the example Notebooks: 

- https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb 
