.. _developer-label:

================
Developer guide
================


Pseudo-distributed sandbox
===========================

.. note::
   These instructions are for users wanting to set up a bare-minimum local deployment of FEDn (without FEDn Studio).
   We here assume practical knowledge of Docker and docker-compose. We recommend all new users of FEDn to start
   by taking the Getting Started tutorial: :ref:`quickstart-label`

During development on FEDn, and when working on own extentions including aggregators and helpers, it is 
useful to have a local development setup of the core FEDn server-side services (controller, combiner, database, object store). 
We provide Dockerfiles and docker-compose template for an all-in-one local sandbox for most examples::

.. code-block::
   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn
   docker compose up -d

This starts up local services for MongoDB, Minio, the API Server, one Combiner. 
You can verify the deployment on localhost using these urls: 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081
  
To run a client in this setup, you can use the CLI to connect to the API Server.

.. code-block::

   pip install -e .
   cd examples/mnist-pytorch
   fedn run client --api-url http://localhost:8092 --local-package

The --local-package flag is used to indicate that the package is available locally in the current directory.
This will enable you to modify the machine learning scripts in the client folder while the client is running.
In otrher words, you don't need to rebuild and upload the compute package every time you make a change. 
Obs that this feature is also available in FEDn Studio.

You can also connect directly to the Combiner (gRPC) without using the API Server (REST-API):

.. code-block::

   fedn run client --combiner=localhost --combiner-port=12080 --local-package

Observe that you need to create an initial model seed.npz in the current directory before starting any new session:

.. code-block::

   fedn run build --path client

Please observe that this local sandbox deployment does not include any of the security and authentication features available in a Studio Project, 
so we will not require authentication of clients (insecure mode) when using the APIClient:  

.. code-block::

   from fedn import APIClient
   client = APIClient(host="localhost", port=8092)
   client.set_active_model("seed.npz")
   client.start_session(rounds=10, timeout=60)



Access message logs and validation data from MongoDB  
------------------------------------------------------
You can access and download event logs and validation data via the API, and you can also as a developer obtain 
the MongoDB backend data using pymongo or via the MongoExpress interface: 

- http://localhost:8081/db/fedn-network/ 

Username and password are found in 'docker-compose.yaml'. 

Access global models   
------------------------------------------------------

You can obtain global model updates from the 'fedn-models' bucket in Minio: 

- http://localhost:9000

Username and password are found in 'docker-compose.yaml'. 

Reset the FEDn deployment   
------------------------------------------------------

To purge all data from a deployment incuding all session and round data, access the MongoExpress UI interface and 
delete the entire ``fedn-network`` collection. Then restart all services. 

Clean up
------------------------------------------------------
You can clean up by running 

.. code-block::

   docker compose down -v

Connecting clients using Docker:
------------------------------------------------------

If you like to run the client in docker as well we have added an extra docker-compose file in the examples folders for this purpose.
This will allow you to run the client in a separate container and connect to the API server using the service name `api-server`:

.. code-block::
   
   docker compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    up



Distributed deployment on a local network
=========================================

You can use different hosts for the various FEDn services. These instructions shows how to set up FEDn on a **local network** using a single workstation or laptop as 
the host for the servier-side components, and other hosts or devices as clients. 

.. note::
   For a secure and production-grade deployment solution over **public networks**, explore the FEDn Studio service at 
   **fedn.scaleoutsystems.com**. 
   
   Alternatively follow this tutorial substituting the hosts local IP with your public IP, open the neccesary 
   ports (see which ports are used in docker-compose.yaml), and ensure you have taken additional neccesary security 
   precautions.
   
**Prerequisites**
-  `One host workstation and atleast one client device`
-  `Python 3.9, 3.10, 3.11 or 3.12 <https://www.python.org/downloads>`__
-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Launch a distributed FEDn Network 
---------------------------------

Start by noting your host's local IP address, used within your network. Discover it by running ifconfig on UNIX or 
ipconfig on Windows, typically listed under inet for Unix and IPv4 for Windows.

Continue by following the standard procedure to initiate a FEDn network, for example using the provided docker-compose template. 
Once the network is active, upload your compute package and seed (for comprehensive details, see the quickstart tutorials).

.. note::
   This guide covers general local networks where server and client may be on different hosts but able to communicate on their private IPs. 
   A common scenario is also to run fedn and the clients on **localhost** on a single machine. In that case, you can replace <host local ip>
   by "127.0.0.1" below.   

Configuring and Attaching Clients
---------------------------------

On your client device, continue with initializing your client. To connect to the host machine we need to ensure we are 
routing the correct DNS to our hosts local IP address. We can do this using the standard FEDn `client.yaml`:

.. code-block::

   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092


We can then run a client using docker by adding the hostname:ip mapping in the docker run command:

.. code-block::

   docker run \
   -v $PWD/client.yaml:<client.yaml file location> \
   <potentiel data pointers>
   —add-host=api-server:<host local ip> \
   —add-host=combiner:<host local ip> \
   <image name> client start -in client.yaml --name client1


Alternatively updating the `/etc/hosts` file, appending the following lines for running naitively:

.. code-block::

   <host local ip>      api-server
   <host local ip>      combiner


.. _auth-label:

Authentication and Authorization (RBAC)
========================================

.. warning:: The FEDn RBAC system is an experimental feature and may change in the future.

FEDn supports Role-Based Access Control (RBAC) for controlling access to the FEDn API and gRPC endpoints. The RBAC system is based on JSON Web Tokens (JWT) and is implemented using the `jwt` package. The JWT tokens are used to authenticate users and to control access to the FEDn API.
There are two types of JWT tokens used in the FEDn RBAC system:
- Access tokens: Used to authenticate access to the FEDn API.
- Refresh tokens: Used to obtain new access tokens when the old ones expire.
 
.. note:: Please note that the FEDn RBAC system is not enabled by default and does not issue JWT tokens. It is used to integrate with external authentication and authorization systems such as FEDn Studio. 

FEDn RBAC system is by default configured with four types of roles:
- `admin`: Has full access to the FEDn API. This role is used to manage the FEDn network using the API client or the FEDn CLI.
- `combiner`: Has access to the /add_combiner endpoint in the API.
- `client`: Has access to the /add_client endpoint in the API and various gRPC endpoint to participate in federated learning sessions.

A full list of the "roles to endpoint" mappings for gRPC can be found in the `fedn/network/grpc/auth.py`. For the API, the mappings are defined using custom decorators defined in `fedn/network/api/auth.py`.

.. note:: The roles are handled by a custom claim in the JWT token called `role`. The claim is used to control access to the FEDn API and gRPC endpoints.

To enable the FEDn RBAC system, you need to set the following environment variables in the controller and combiner:

Authentication Environment Variables
-------------------------------------

.. line-block::

     **FEDN_JWT_SECRET_KEY**
      - **Type:** str
      - **Required:** yes
      - **Default:** None
      - **Description:** The secret key used for JWT token encryption.

     **FEDN_JWT_ALGORITHM**
      - **Type:** str
      - **Required:** no
      - **Default:** "HS256"
      - **Description:** The algorithm used for JWT token encryption.

     **FEDN_AUTH_SCHEME**
      - **Type:** str
      - **Required:** no
      - **Default:** "Token"
      - **Description:** The authentication scheme used in the FEDn API and gRPC interceptors.

Additional Environment Variables
--------------------------------

For further flexibility, you can also set the following environment variables:

.. line-block::

     **FEDN_CUSTOM_URL_PREFIX**
      - **Type:** str
      - **Required:** no
      - **Default:** None
      - **Description:** Add a custom URL prefix used in the FEDn API, such as /internal or /v1.

     **FEDN_AUTH_WHITELIST_URL**
      - **Type:** str
      - **Required:** no
      - **Default:** None
      - **Description:** A URL pattern to the API that should be excluded from the FEDn RBAC system. For example, /internal (to enable internal API calls).

     **FEDN_JWT_CUSTOM_CLAIM_KEY**
      - **Type:** str
      - **Required:** no
      - **Default:** None
      - **Description:** The custom claim key used in the JWT token.

     **FEDN_JWT_CUSTOM_CLAIM_VALUE**
      - **Type:** str
      - **Required:** no
      - **Default:** None
      - **Description:** The custom claim value used in the JWT token.

Client Environment Variables
-----------------------------

For the client, you need to set the following environment variables:

.. line-block::

     **FEDN_AUTH_REFRESH_TOKEN_URI**
      - **Type:** str
      - **Required:** no
      - **Default:** None
      - **Description:** The URI used to obtain new access tokens when the old ones expire.

     **FEDN_AUTH_REFRESH_TOKEN**
      - **Type:** str
      - **Required:** no
      - **Default:** None
      - **Description:** The refresh token used to obtain new access tokens when the old ones expire.

     **FEDN_AUTH_SCHEME**
      - **Type:** str
      - **Required:** no
      - **Default:** "Token"
      - **Description:** The authentication scheme used in the FEDn API and gRPC interceptors.

You can use `--token` flags in the FEDn CLI to set the access token.

.. meta::
   :description lang=en:
      During development on FEDn, and when working on own extentions including aggregators and helpers, it is useful to have a local development setup.
   :keywords: Federated Learning, Developer guide, Federated Learning Framework, Federated Learning Platform, FEDn, Scaleout Systems
   