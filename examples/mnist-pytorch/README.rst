Quickstart Tutorial PyTorch (MNIST)
-------------

This classic example of hand-written text recognition is well suited as a lightweight test when developing on FEDn in pseudo-distributed mode. 
A normal high-end laptop or a workstation should be able to sustain a few clients. 
The example automates the partitioning of data and deployment of a variable number of clients on a single host. 
We here assume working experience with containers, Docker and docker-compose. 
   
Prerequisites
-------------

-  `Python 3.8, 3.9 or 3.10 <https://www.python.org/downloads>`__
-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Quick start
-----------

Clone this repository, locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch

Start a pseudo-distributed FEDn network using docker-compose:

.. code-block::

   docker-compose -f ../../docker-compose.yaml up

This starts up the needed backend services MongoDB and Minio, the API Server and one Combiner. 
You can verify the deployment using these urls: 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081

Next, we will prepare the client. A key concept in FEDn is the compute package - 
a code bundle that contains entrypoints for training and (optionally) validating a model update on the client. 

Locate into 'examples/mnist-pytorch' and familiarize yourself with the project structure. The entrypoints
are defined in 'client/entrypoint'. The dependencies needed in the client environment are specified in 
'requirements.txt'. For convenience, we have provided utility scripts to set up a virtual environment.    

Start by initializing a virtual enviroment with all of the required dependencies for this project.

.. code-block::

   bin/init_venv.sh

Next create the compute package and a seed model:

.. code-block::

   bin/build.sh

You should now have a file 'package.tgz' and 'seed.npz' in the project folder. 

Next we prepare the local dataset. For this we download MNIST data and make data partitions: 

Download the data:

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


Automate experimentation with several clients:  
-----------

Now that you have an understanding of the main components of FEDn, you can use the provided docker-compose templates to automate deployment of FEDn and clients. 
To start the network and attach 4 clients: 

.. code-block::

   docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up --scale client=4 


Access logs and validation data from MongoDB  
-----------
You can access and download event logs and validation data via the API, and you can also as a developer obtain 
the MongoDB backend data using pymongo or via the MongoExpress interface: 

- http://localhost:8081/db/fedn-network/ 

The credentials are as set in docker-compose.yaml in the root of the repository. 

Access model updates  
-----------

You can obtain model updates from the 'fedn-models' bucket in Minio: 

- http://localhost:9000


Clean up
-----------
You can clean up by running 

.. code-block::

   docker-compose down
