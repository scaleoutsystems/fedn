.. _server-functions:

Modifying Server Functionality
==============================

Scaleout Edge provides an interface where you can implement your own server-side logic directly into your server by utilizing the ``ServerFunctions`` class. This enables advanced customization of the server's behavior while working with Scaleout Edge.
You can for example implement custom client selection logic, adjust hyperparameters, or implement a custom aggregation algorithm. See https://www.youtube.com/watch?v=Rnfhfqy_Tts for information in video format.

Requirements for ``ServerFunctions`` Implementation
----------------------------------------------------

The ``ServerFunctions`` class has specific requirements for proper instantiation at the server:

1. **Class Name**: The implemented class must be named ``ServerFunctions``.
2. **Allowed Imports**: Only a pre-defined list of Python packages is available for use within a ``ServerFunctions`` implementation for compatibility and security reasons. You can find the allowed packages at:

   :py:mod:`scaleout-client.scaleout.network.combiner.hooks.allowed_imports`.

Overridable Methods
-------------------

The ``ServerFunctions`` class provides three methods that can optionally be overridden. If you choose not to override one or several of these, Scaleout Edge will execute its default behavior for that functionality.

The base class defining these methods and their types is:

:py:mod:`scaleout-util.scaleoututil.serverfunctions.serverfunctionsbase.ServerFunctionsBase`.

The methods available for customization are:

1. **client_selection(client_ids: List[str]) -> List[str]**:
   Called at the beginning of a round to select clients.

2. **client_settings(global_model: List[np.ndarray]) -> dict**:
   Called before sending the global model to configure client-specific settings.

3. **aggregate(previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]**:
   Called after receiving client updates to aggregate them into a new global model.

Example: Customizing Server Functions
-------------------------------------

Below is an example of how to implement custom server functionality in a ``ServerFunctions`` class.

.. code-block:: python

    from fedn.common.log_config import logger
    from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

    class ServerFunctions(ServerFunctionsBase):
        def __init__(self) -> None:
            self.round = 0  # Keep track of training rounds
            self.lr = 0.1  # Initial learning rate

        def client_selection(self, client_ids: List[str]) -> List[str]:
            # Select up to 10 random clients
            return random.sample(client_ids, min(len(client_ids), 10))

        def client_settings(self, global_model: List[np.ndarray]) -> dict:
            # Adjust the learning rate every 10 rounds
            if self.round % 10 == 0:
                self.lr *= 0.1
            self.round += 1
            return {"learning_rate": self.lr}

        def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
            # Implement a weighted FedAvg aggregation
            weighted_sum = [np.zeros_like(param) for param in previous_global]
            total_weight = 0
            for client_id, (client_parameters, metadata) in client_updates.items():
                num_examples = metadata.get("num_examples", 1)
                total_weight += num_examples
                for i, param in enumerate(client_parameters):
                    weighted_sum[i] += param * num_examples
            logger.info("Models aggregated")
            return [param / total_weight for param in weighted_sum]

Using ``ServerFunctions`` in Scaleout Edge
------------------------------------------

To use your custom ``ServerFunctions`` code in Scaleout Edge, follow these steps:

1. **Prepare Your Environment**:

   Ensure you have an API token for your project. Retrieve it from the "Settings" page in your Scaleout Edge UI and add it to your environment:

   .. code-block:: bash

       export SCALEOUT_AUTH_TOKEN=<your_access_token>

2. **Connect Using the API Client**:

   Connect to your Scaleout Edge project using the ``APIClient``. Replace ``<controller-host>`` with the address found on the Scaleout Edge dashboard.

   .. code-block:: python

       from scaleout-client import APIClient
       client = APIClient(host="<controller-host>", secure=True, verify=True)

3. **Start a Session with ``ServerFunctions``**:

   After uploading a model seed, compute package, and connecting clients, you can start a session with your custom ``ServerFunctions`` class:

   .. code-block:: python

       from server_functions import ServerFunctions
       client.start_session(server_functions=ServerFunctions)

4. **Monitor Logs**:

   Logs from your ``ServerFunctions`` implementation can be viewed on the Scaleout Edge dashboard under the "Logs" section.

Notes
-----

- **Documentation**: Refer to the full APIClient documentation for more details on connecting to your project:

  https://docs.scaleoutsystems.com/en/stable/apiclient.html

This modular interface enables you to integrate your specific server-side logic into your Scaleout Edge federated learning pipeline.
