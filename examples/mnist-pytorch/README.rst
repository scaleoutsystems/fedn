Quickstart Tutorial PyTorch (MNIST)
-------------

This classic example of hand-written text recognition is well suited as a lightweight test when developing on FEDn in pseudo-distributed mode. 
A normal high-end laptop or a workstation should be able to sustain a few clients. 
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

Clone this repository, locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch

Create the package using script:

.. code-block::

   bin/build.sh

This should create a file 'package.tgz' in the project folder.

Upload the package to FEDn Studio project. 

Start the client using the client.yaml file from FEDn Studio.

.. code-block::
  pip install fedn 
  export FEDN_AUTH_SCHEME=Bearer
  export FEDN_PACKAGE_EXTRACT_DIR=package
  fedn run client -in client.yaml --name client1 --secure=True --force-ssl


Upload the initial model to FEDn Studio project. The seed.npz file is created when you run start the client and is found in ./data/models/seed.npz

The default traning and test data is found in ./data/clients/1/mnist.pt and can be changed to other partitions by exporting the environment variable FEDN_DATA_PATH.
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

   bin/build.sh

You should now have a file 'package.tgz'. 

You are now ready to use the API to initialize the system with the package.

Obs - After you have uploaded the package, you need to fetch the initial model (seed.npz) from client container:

.. code-block::

   docker cp fedn-client-1:/app/package/data/models/seed.npz .

- Follow the example in the `Jupyter Notebook <https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb>`__


Automate experimentation with several clients:  
----------------------------------------------

If you want to scale the number of clients, you can do so by running the following command:

.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up --scale client=4 


Access logs and validation data from MongoDB  
--------------------------------------------
You can access and download event logs and validation data via the API, and you can also as a developer obtain 
the MongoDB backend data using pymongo or via the MongoExpress interface: 

- http://localhost:8081/db/fedn-network/ 

The credentials are as set in docker-compose.yaml in the root of the repository. 

Access model updates  
--------------------

You can obtain model updates from the 'fedn-models' bucket in Minio: 

- http://localhost:9000


Clean up
-------
You can clean up by running 

.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml down -v
