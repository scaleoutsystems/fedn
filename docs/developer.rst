.. _developer-label:

================
Developer guide
================


Pseudo-distributed sandbox
===========================

.. note::
   These instructions are for users wanting to set up a bare-minimum local development deployment of FEDn (without FEDn Studio).
   We recommend all new users of FEDn to start by taking the Getting Started tutorial: :ref:`quickstart-label`

During development on FEDn, and when working on own extentions including aggregators and helpers, it is 
useful to have a simple, local development setup of the core FEDn server-side services: api-server, combiner, database, and object store. 
We provide Dockerfiles and docker-compose template for an all-in-one sandbox: 

On the server: 

.. code-block::

   docker compose up

This starts up containers for MongoDB, Minio, the FEDn API Server and a combiner. 
You can verify the deployment on the server using these urls (here assuming deployment on localhost): 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081

To start clients, we need to make sure that we can resolver the names "api-server" and "combiner". 
This can be accomplished by editing the  `/etc/hosts` file. Add the following lines, where you replace `<host ip>` with the local IP of the server (or localhost):

.. code-block::

   <host ip>      api-server
   <host ip>      combiner

To connect to the api-server and set the package and seed model, you can use the following code snippet:

.. code-block::

   from fedn import APIClient
   client = APIClient(host="api-server", port=8092)
   client.set_active_package("package.tgz", helper="numpyhelper", name="my-package")
   client.set_active_model("seed.npz")

.. note::
   For a secure and production-grade deployment solution over **public networks**, explore the FEDn Studio service at 
   **fedn.scaleoutsystems.com**. 

Access logs and validation data in MongoDB  
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

   docker-compose down -v
   
.. _auth-label:

Authentication and Authorization (RBAC)
========================================

.. warning:: The FEDn RBAC system is an experimental feature and may change in the future.

FEDn supports Role-Based Access Control (RBAC) for controlling access to the FEDn API and gRPC endpoints. The RBAC system is based on JSON Web Tokens (JWT) and is implemented using the `jwt` package. The JWT tokens are used to authenticate users and to control access to the FEDn API.
There are two types of JWT tokens used in the FEDn RBAC system:
- Access tokens: Used to authenticate users and to control access to the FEDn API.
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
   