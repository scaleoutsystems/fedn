.. _local_compute-label:

Local Compute: Running Scaleout Edge Clients With Local Code
============================================================

In a standard Scaleout Edge workflow, the server distributes a *compute package*
(containing the client-side training and validation code) to connected clients.
This enables rapid experimentation and easy onboarding, but it also means the
server decides **what code will be executed on the client device**, which may be
undesirable or restricted in regulated or production environments.

**Local compute** is the alternative execution mode where the client **decides its
own training code locally**, without downloading or executing server-provided
packages. Instead, the code remains fully controlled by the client owner, and
the federated workflow (training rounds, orchestration, and model exchange)
still operates through the Scaleout Edge network.

Local compute is enabled using the ``--local-package`` flag when starting a
client, for example:

.. code-block:: bash

    scaleout client start --init client_config.yaml --local-package


Advantages of local compute include:

- Full control and auditability of executed code
- Improved security posture for production or regulated environments
- No need to package, upload, or manage compute code on the server
- Easy to prototype in notebooks or standalone Python files (e.g. Google Colab)

Follow the tutorial in `this notebook <https://github.com/scaleoutsystems/scaleout-client/python/examples/api-tutorials/Local_Compute_Example.ipynb>`_
to get started.


Steps involved
--------------
1. Create and upload a seed model to Scaleout Edge
2. Define all client-side training (and optional validation) logic locally
3. Start the client with ``--local-package`` and connect it to your project
4. Start a training session in Scaleout Edge and let your local code run
