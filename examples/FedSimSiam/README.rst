FEDn Project: MNIST (PyTorch)
-----------------------------

This is an example FEDn Project that runs the federated self-supervised learning algorithm FedSimSiam on 
the CIFAR-10 dataset. This is a standard example often used for benchmarking. To be able to run this example, you 
need to have GPU access. 

   **Note: We recommend all new users to start by following the Quickstart Tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html** 

Prerequisites
-------------

-  `Python 3.8, 3.9, 3.10 or 3.11 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__   
-  Change the dependencies in the 'client/python_env.yaml' file to match your cuda version.

Creating the compute package and seed model
-------------------------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch

Create the compute package:

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in a global model trail):

.. code-block::

   fedn run build --path client

This will create a seed model called 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  

Using FEDn Studio
-----------------

Follow the instructions to register for FEDN Studio and start a project (https://fedn.readthedocs.io/en/stable/studio.html).

In your Studio project:

- Go to the 'Sessions' menu, click on 'New session', and upload the compute package (package.tgz) and seed model (seed.npz).
- In the 'Clients' menu, click on 'Connect client' and download the client configuration file (client.yaml)
- Save the client configuration file to the FedSimSiam example directory (fedn/examples/FedSimSiam)

To connect a client, run the following command in your terminal:

.. code-block::

   fedn client start -in client.yaml --secure=True --force-ssl


Running the example
-------------------

After everything is set up, go to 'Sessions' and click on 'New Session'. Click on 'Start run' and the example will execute. You can follow the training progress on 'Events' and 'Models', where you 
can monitor the training progress. The monitoring is done using a kNN classifier that is fitted on the feature embeddings of the training images that are obtained by
FedSimSiam's encoder, and evaluated on the feature embeddings of the test images. This process is repeated after each training round.

This is a common method to track FedSimSiam's training progress, as FedSimSiam aims to minimize the distance between the embeddings of similar images.
A high accuracy implies that the feature embeddings for images within the same class are indeed close to each other in the
embedding space, i.e., FedSimSiam learned useful feature embeddings.


Running FEDn in local development mode:
---------------------------------------

Follow the steps above to install FEDn, generate 'package.tgz' and 'seed.tgz'.

Start a pseudo-distributed FEDn network using docker-compose:
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

Upload the package and seed model to FEDn controller using the APIClient:

.. code-block::

   from fedn import APIClient
   client = APIClient(host="localhost", port=8092)
   client.set_active_package("package.tgz", helper="numpyhelper")
   client.set_active_model("seed.npz")


You can now start a training session with 100 rounds using the API client:

.. code-block::

   client.start_session(rounds=100)

Clean up 
--------

You can clean up by running

.. code-block::

   docker-compose \
   -f ../../docker-compose.yaml \
   -f docker-compose.override.yaml \
   down -v
