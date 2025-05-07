FEDn Project: Server functions toy example
-----------------------------


For details on the functionality of server-functions see either the file server_functions.py, 
the docs https://docs.scaleoutsystems.com/en/stable/serverfunctions.html or the youtube video 
https://www.youtube.com/watch?v=Rnfhfqy_Tts.

To apply server-functionality in studio first connect to your project through the APIClient.

See https://docs.scaleoutsystems.com/en/stable/apiclient.html for more information.

When connected to the project API you can start sessions with your supplied server functions.

To test your server functions locally you can use the function `fedn.network.combiner.hooks.serverfunctionstest.test_server_functions`, 
see `test_locally.py` for a running example.

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

Logs from the server functions code are visible from the studio dashboard logs page.
