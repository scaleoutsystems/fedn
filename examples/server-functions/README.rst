FEDn Project: Server functions toy example
-----------------------------

Beta useage will be available by request from FEDn 0.20.0 and onward.

See server_functions.py for details on server-functionality can be supplied.

To apply server-functionality in studio first connect to your project through the APIClient.

See https://docs.scaleoutsystems.com/en/stable/apiclient.html for more information.

When connected to the project API you can start sessions with your supplied server functions.

Full commands to run through the API client:

Get your token from the settings page in your studio project and add it in your system environment.

.. code-block::
    export FEDN_AUTH_TOKEN=<access token>

Connect through the APIClient from a python instance, you can find your controller host on the studio Dashboard page.

.. code-block::
    from fedn import APIClient
    client = APIClient(host="<controller-host>", secure=True, verify=True)

Start a session with your ServerFunctions code (assuming you have uploaded a model seed, compute package and have connected clients).

.. code-block::
    from server_functions import ServerFunctions
    client.start_session(server_functions=ServerFunctions)