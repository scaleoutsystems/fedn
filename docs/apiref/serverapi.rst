Server API
==========

This section documents the server-side API exposed by the Scaleout Edge control
plane. These endpoints are used by clients, combiners, and external integrations
to interact with the Scaleout Edge network. The API includes operations for
project management, training orchestration, model handling, metrics, telemetry,
and system-level metadata.

Authentication and Control
--------------------------

Endpoints related to authentication, authorization, and high-level control of
the federated network.

.. automodule:: auth_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: control_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: attribute_routes
   :members:
   :undoc-members:
   :show-inheritance:


Clients and Combiners
---------------------

Endpoints for managing clients, combiners, and their runtime state in the
federated network.

.. automodule:: client_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: combiner_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: status_routes
   :members:
   :undoc-members:
   :show-inheritance:


Sessions, Rounds, and Runs
--------------------------

Endpoints that control training sessions, orchestration rounds, and execution
runs.

.. automodule:: session_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: round_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: run_routes
   :members:
   :undoc-members:
   :show-inheritance:


Models, Packages, and Predictions
---------------------------------

Endpoints for handling model artifacts, compute packages, predictions, and
validation.

.. automodule:: model_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: package_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: prediction_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: validation_routes
   :members:
   :undoc-members:
   :show-inheritance:


Metrics, Telemetry, and Helpers
-------------------------------

Endpoints for metrics, telemetry, and supporting helper operations.

.. automodule:: metric_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: telemetry_routes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: helper_routes
   :members:
   :undoc-members:
   :show-inheritance:
