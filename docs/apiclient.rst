.. _apiclient-label:

APIClient
=========

FEDn comes with an *APIClient* - a Python3 library that can be used to interact with FEDn programmatically. 

**Installation**

The APIClient is available as a Python package on PyPI, and can be installed using pip:

.. code-block:: bash
   
   $ pip install fedn

**Initialize the APIClient**

The FEDn REST API is available at <controller-host>/api/v1/. To access this API you need the url to the controller-host, as well as an admin API token. The controller host can be found in the project dashboard (top right corner).
To obtain an admin API token, navigate to the "Settings" tab in your Studio project and click on the "Generate token" button. Copy the 'access' token and use it to access the API using the instructions below. 


.. code-block:: python

   >>> from fedn import APIClient
   >>> client = APIClient(host="<controller-host>", token="<access-token>", secure=True, verify=True)

Alternatively, the access token can be sourced from an environment variable. 

.. code-block:: bash
   $ export FEDN_AUTH_TOKEN=<access-token>

Then passing a token as an argument is not required. 

.. code-block:: python

   >>> from fedn import APIClient
   >>> client = APIClient(host="<controller-host>", secure=True, verify=True)


**Set active package and seed model**

The active package can be set using the following code snippet:

.. code-block:: python
   
   client.set_active_package(path="path/to/package.tgz", helper="numpyhelper")

To set the initial seed model, you can use the following code snippet:

.. code-block:: python
   
   client.set_active_model(path="path/to/seed.npz")

**Start a training session**

Once the active package and seed model are set, you can connect clients to the network and start training models. The following code snippet starts a traing session:

.. code-block:: python
   
   session = client.start_session(id="session_name")

**List data**

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


For more information on how to use the APIClient, see the :py:mod:`fedn.network.api.client`, and the example `Notebooks <https://github.com/scaleoutsystems/fedn/tree/master/examples/notebooks>`_. 
