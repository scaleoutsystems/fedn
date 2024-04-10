Quickstart Tutorial PyTorch (MNIST)
-----------------------------------

This is an example FEDn Project based on the classic hand-written text recognition dataset MNIST.
The project is well suited as a lightweight test when developing on FEDn in pseudo-distributed mode. 
A normal high-end laptop or a workstation (CPU only or GPU) should be able to sustain a few clients. 
The example automates the partitioning of data and deployment of a variable number of clients on a single host. 
We here assume working experience with containers, Docker and docker-compose. 
   
Prerequisites
-------------

Using FEDn Studio:

-  `Python 3.8, 3.9, 3.10 or 3.11 <https://www.python.org/downloads>`__

If using self-managed with docker-compose:

-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Quick start with FEDn Studio
----------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch

Create the compute package (compress the 'client' folder):

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in the global model trail):

.. code-block::

   fedn run build --path client

Upload the package and seed model to the FEDn Studio project on the "Session" page. 

In Studio, go to "Clients" and download a new client configuration file (contains the access token). 
Then, start the client using the client.yaml file:

.. code-block::

   export FEDN_AUTH_SCHEME=Bearer
   export FEDN_PACKAGE_EXTRACT_DIR=package
   fedn run client -in client.yaml --secure=True --force-ssl

The default traning and test data is for this example downloaded and split direcly by the client when it starts up. 
The data will be found in package/data/clients/1/mnist.pt and can be changed to other partitions by exporting the environment variable FEDN_DATA_PATH.
For example, to use the second partiton:

.. code-block::

   export FEDN_DATA_PATH=data/clients/2/mnist.pt

The default split into 2 partitions can be changed in client/data.py.

Quick start with docker-compose
-------------------------------

Clone this repository, locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch

Start a pseudo-distributed FEDn network using docker-compose:

.. code-block::

   docker compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    up -d

This starts up the needed backend services MongoDB and Minio, the API Server and one Combiner. As well as two clients. 
You can verify the deployment using these urls: 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081

To check the client output logs, run:

.. code-block::

   docker logs fedn-client-1

It should be waiting for the configuration of the package.

Create the package (compress client folder):

.. code-block::

   tar -czvf package.tgz client 

You should now have a file 'package.tgz'. 

You are now ready to use the API to initialize the system with the package.

Obs - After you have uploaded the package, you need to fetch the initial model (seed.npz) from client container:

.. code-block::

   bin/get_data


Split the data in 10 partitions:

.. code-block::

   bin/split_data --n_splits=10

Data partitions will be generated in the folder 'data/clients'.  

FEDn relies on a configuration file for the client to connect to the server. Create a file called 'client.yaml' with the follwing content:

.. code-block::

   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092

Make sure to move the file ``client.yaml`` to the root of the examples/mnist-pytorch folder.
To connect a client that uses the data partition ``data/clients/1/mnist.pt`` and the config file ``client.yaml`` to the network, run the following docker command:

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -v $PWD/data/clients/1:/var/data \
     -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.pt \
     --network=fedn_default \
     ghcr.io/scaleoutsystems/fedn/fedn:master-mnist-pytorch run client -in client.yaml --name client1

Observe the API Server logs and combiner logs, you should see the client connecting and entering into a state asking for a compute package. 

In a separate terminal, start a second client using the data partition 'data/clients/2/mnist.pt':

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -v $PWD/data/clients/2:/var/data \
     -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.pt \
     --network=fedn_default \
     ghcr.io/scaleoutsystems/fedn/fedn:master-mnist-pytorch run client -in client.yaml --name client2
 
You are now ready to use the API to initialize the system with the compute package and seed model, and to start federated training. 

- Follow the example in the `Jupyter Notebook <https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb>`__


Automate experimentation with several clients  
-----------------------------------------------

If you want to scale the number of clients, you can do so by running the following command:

.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up --scale client=4 


Access logs and validation data from MongoDB  
---------------------------------------------
You can access and download event logs and validation data via the API, and you can also as a developer obtain 
the MongoDB backend data using pymongo or via the MongoExpress interface: 

- http://localhost:8081/db/fedn-network/ 

The credentials are as set in docker-compose.yaml in the root of the repository. 

Access model updates  
---------------------

You can obtain model updates from the 'fedn-models' bucket in Minio: 

- http://localhost:9000


Clean up
--------
You can clean up by running 

.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml down -v
