User interfaces
===============

FEDn comes with an *APIClient* and a *Dashboard* for interacting with the FEDn network. The APIClient is a Python3 library that can be used to interact with the FEDn network programmatically. 
The Dashboard is a web-based user interface that can be used to interact with the FEDn network through a web browser.

APIClient
--------------
The APIClient is a Python3 library that can be used to interact with the FEDn network programmatically. The APIClient is available as a Python package on PyPI, and can be installed using pip:

.. code-block:: bash
   
   $ pip install fedn


To initialize the APIClient, you need to provide the hostname and port of the FEDn API server. The default port is 8092. The following code snippet shows how to initialize the APIClient:

.. code-block:: python
   
   from fedn import APIClient
   client = APIClient("localhost", 8092)

For more information on how to use the APIClient, see the :py:mod:`fedn.network.api.client`.

Dashboard
--------------
The Dashboard is a web-based user interface that can be used to interact with the FEDn network through a web browser. The Dashboard is available as a Docker image, and can be run using the following command:

.. code:: bash
   
   $ docker-compose up -d dashboard

OBS! If you have followed any of the examples, the dashboard will already be running! 
The Dashboard is now available at http://localhost:8090. If no compute package has been configured, the Dashboard will ask you to upload a compute package. 
A compute package is a zip file containing the ML code that will be executed on the clients. 
For more information on how to create a compute package, see the :ref:`tutorial-label`. After uploading a compute package, you will also need to upload an initial model. This initial model is
usually the initial weights for the model that will be trained. You can then navigate to the Control Panel to start a training session. 
