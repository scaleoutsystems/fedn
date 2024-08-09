.. _developer-label:

Local development and deployment
================================

.. note::
   These instructions are for users wanting to set up a local development deployment of FEDn (i.e. without FEDn Studio).
   This requires practical knowledge of Docker and docker-compose. 

Running the FEDn development sandbox (docker-compose)
------------------------------------------------------

During development on FEDn, and when working on own aggregators/helpers, it is 
useful to have a local development setup of the core FEDn services (controller, combiner, database, object store). 
For this, we provide Dockerfiles and docker-compose template. 

To start a development sandbox for FEDn using docker-compose:

.. code-block::

   docker compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    up

This starts up local services for MongoDB, Minio, the API Server, one Combiner and two clients. 
You can verify the deployment using these urls: 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081

This setup does not include the security features of Studio, and thus will not require authentication of clients. 
To use the APIClient to test a compute package and seed model against a local FEDn deployment: 

.. code-block::

   from fedn import APIClient
   client = APIClient(host="localhost", port=8092)
   client.set_active_package("package.tgz", helper="numpyhelper")
   client.set_active_model("seed.npz")


To connect a native FEDn client, you need to make sure that the combiner service can be resolved using the name "combiner". 
One way to achieve this is to edit your '/etc/hosts' and add a line '127.0.0.1  	combiner'. 

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

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml down -v


Connecting clients using Docker:
------------------------------------------------------

For convenience, we distribute a Docker image hosted on ghrc.io with FEDn preinstalled. For example, to start a client for the MNIST PyTorch example using Docker
and FEDN 0.10.0, run this from the example folder:   

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -e FEDN_PACKAGE_EXTRACT_DIR=package \
     -e FEDN_NUM_DATA_SPLITS=2 \
     -e FEDN_DATA_PATH=/app/package/data/clients/1/mnist.pt \
     ghcr.io/scaleoutsystems/fedn/fedn:0.10.0 run client -in client.yaml --force-ssl --secure=True


Self-managed distributed deployment
------------------------------------------------------

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
-  `Python 3.8, 3.9, 3.10 or 3.11 <https://www.python.org/downloads>`__
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
   <image name> run client -in client.yaml --name client1


Alternatively updating the `/etc/hosts` file, appending the following lines for running naitively:

.. code-block::

   <host local ip>      api-server
   <host local ip>      combiner
