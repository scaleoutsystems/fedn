.. _apiclient-label:

Using the FEDn API Client
====================

FEDn comes with an *APIClient* - a Python3 library that can be used to interact with FEDn programmatically. 
In this tutorial we show how to use the APIClient to initialize the server-side with the compute package and seed models, 
run and control training sessions, use different aggregators, and to retrieve models and metrics. 

We assume a basic understanding of the FEDn framework, i.e. that the user has taken the :ref:`quickstart-label` tutorial.

**Installation**

The APIClient is available as a Python package on PyPI, and can be installed using pip:

.. code-block:: bash
   
   $ pip install fedn

**Initialize the APIClient to a FEDn Studio project**

The FEDn REST API is available at <controller-host>/api/v1/. To access this API you need the url to the controller-host, as well as an admin API token. The controller host can be found in the project dashboard (top right corner).
To obtain an admin API token, 

#. Navigate to the "Settings" tab in your Studio project and click on the "Generate token" button. 
#. Copy the 'access' token and use it to access the API using the instructions below. 

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

**Set the active compute package and seed model**

To set the active compute package in the FEDn Studio Project:

.. code:: python

   >>> client.set_active_package("package.tgz", helper="numpyhelper")
   >>> client.set_active_model("seed.npz")

**Start a training session**

Once the active package and seed model are set, you can connect clients to the network and start training models. To run a training session
using the default aggregator (FedAvg):

.. code:: python

   >>> ...
   >>> client.start_session(id="test-session", rounds=3)
   # Wait for training to complete, when controller is idle:
   >>> client.get_controller_status()
   # Show model trail:
   >>> models = client.get_model_trail()
   # Show performance of latest global model:
   >>> model_id = models[-1]['model']
   >>> validations = client.get_validations(model_id=model_id)

To run a session using the FedAdam aggregator using custom hyperparamters: 

.. code-block:: python

   >>> session_id = "experiment_fedadam"

   >>> session_config = {
                     "helper": "numpyhelper",
                     "id": session_id,
                     "aggregator": "fedopt",
                     "aggregator_kwargs": {
                           "serveropt": "adam",
                           "learning_rate": 1e-2,
                           "beta1": 0.9,
                           "beta2": 0.99,
                           "tau": 1e-4
                           },
                     "model_id": seed_model['model'],
                     "rounds": 10
                  }

   >>> result_fedadam = client.start_session(**session_config)

**Download a global model**

To download a global model and write it to file:

.. code:: python

   >>> ...
   >>> client.download_model("<model-id>", path="model.npz")

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

To list all sessions: 
.. code-block:: python
   
   >>> sessions = client.get_sessions()

To get a specific session:

.. code-block:: python
   
   >>> session = client.get_session(id="session_name")

For more information on how to use the APIClient, see the :py:mod:`fedn.network.api.client`, and the collection of example Jupyter Notebooks:
 
- `API Example <https://github.com/scaleoutsystems/fedn/tree/master/examples/notebooks>`_  . 


.. meta::
   :description lang=en:
      FEDn comes with an APIClient - a Python3 library that can be used to interact with FEDn programmatically.
   :keywords: Federated Learning, APIClient, Federated Learning Framework, Federated Learning Platform, FEDn, Scaleout Systems
   
