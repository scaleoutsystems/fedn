Hugging Face Transformer Example
--------------------------------

This is an example project that demonstrates how one can make use of the Hugging Face Transformers library in FEDn.
In this example, a pre-trained BERT-tiny model from Hugging Face is fine-tuned to perform spam detection 
on the Enron spam email dataset.

Email communication often contains personal and sensitive information, and privacy regulations make it 
impossible to collect the data to a central storage for model training.
Federated learning is a privacy preserving machine learning technique that enables the training of models on decentralized data sources.
Fine-tuning large language models (LLMs) on various data sources enhances both accuracy and generalizability.
In this example, the Enron email spam dataset is split among two clients. The BERT-tiny model is fine-tuned on the client data using 
federated learning to predict whether an email is spam or not.
Execute the following steps to run the example:

Prerequisites
-------------

Using FEDn Studio:

-  `Python 3.8, 3.9, 3.10 or 3.11 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__   

If using pseudo-distributed mode with docker-compose:

-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Creating the compute package and seed model
-------------------------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/huggingface

Create the compute package:

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in a global model trail):

.. code-block::

   fedn run build --path client

This will create a seed model called 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  



Using FEDn Studio (recommended)
-------------------------------

Follow the instructions to register for FEDN Studio and start a project (https://fedn.readthedocs.io/en/stable/studio.html).

In your Studio project:

- Go to the 'Sessions' menu, click on 'New session', and upload the compute package (package.tgz) and seed model (seed.npz).
- In the 'Clients' menu, click on 'Connect client' and download the client configuration file (client.yaml)
- Save the client configuration file to the huggingface example directory (fedn/examples/huggingface)

To connect a client, run the following command in your terminal:

.. code-block::

   fedn run client -in client.yaml --secure=True --force-ssl
   

Alternatively, if you prefer to use Docker, run the following:

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   -e CLIENT_NUMBER=0 \
   -e FEDN_PACKAGE_EXTRACT_DIR=package \
   ghcr.io/scaleoutsystems/fedn/fedn:0.9.0 run client -in client.yaml --secure=True --force-ssl


Running the example
-------------------

After everything is set up, go to 'Sessions' and click on 'New Session'. Click on 'Start run' and the example
will execute. You can follow the training progress on 'Events' and 'Models', where you can view the calculated metrics.



Running FEDn in local development mode:
---------------------------------------

Create the compute package and seed model as explained above. Then run the following command:


.. code-block::

   docker-compose \
   -f ../../docker-compose.yaml \
   -f docker-compose.override.yaml \
   up


This starts up local services for MongoDB, Minio, the API Server, one Combiner and two clients. You can verify the deployment using these urls:

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081


Upload the package and seed model to FEDn controller using the APIClient:

.. code-block::

    from fedn import APIClient
    client = APIClient(host="localhost", port=8092)
    client.set_active_package("package.tgz", helper="numpyhelper")
    client.set_active_model("seed.npz")


You can now start a training session with 5 rounds (default) using the API client:

.. code-block::

    client.start_session()

Clean up 
--------

You can clean up by running 

.. code-block::

   docker-compose \
   -f ../../docker-compose.yaml \
   -f docker-compose.override.yaml \
   down -v
