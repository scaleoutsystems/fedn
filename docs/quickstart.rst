Quick Start
===========

Clone this repository, locate into it and start a pseudo-distributed FEDn network using docker-compose:

.. code-block::

   docker-compose up 


This starts up the needed backend services MongoDB and Minio, the API Server and one Combiner. 
You can verify the deployment using these urls: 

- API Server: http://localhost:8092/get_controller_status
- Minio: http://localhost:9000
- Mongo Express: http://localhost:8081

.. warning:: 
   The FEDn network is configured to use a local Minio and MongoDB instances for storage. This is not suitable for production, but is fine for testing.

.. note::
    To programmatically interact with the FEDn network use the APIClient.
    Install the FEDn via pip:

    .. code-block:: bash
       
       $ pip install fedn
       # or from source
       $ cd fedn
       $ pip install . 

Next, we will prepare the client. A key concept in FEDn is the compute package - 
a code bundle that contains entrypoints for training and (optionally) validating a model update on the client. 

Locate into 'examples/mnist-pytorch' and familiarize yourself with the project structure. The entrypoints
are defined in 'client/entrypoint'. The dependencies needed in the client environment are specified in 
``requirements.txt``. For convenience, we have provided utility scripts to set up a virtual environment.   

Start by initializing a virtual enviroment with all of the required dependencies for this project.

.. code-block:: python

   bin/init_venv.sh

Now create the compute package and an initial model:

.. code-block::

   bin/build.sh


Upload the compute package and seed model to FEDn:

.. code:: python

   >>> from fedn import APIClient
   >>> client = APIClient(host="localhost", port=8092)
   >>> client.set_package("package.tgz", helper="numpyhelper")
   >>> client.set_initial_model("seed.npz")      

The next step is to configure and attach clients. For this we need to download data and make data partitions: 

Download the data:

.. code-block::

   bin/get_data


Split the data in 2 parts for the clients:

.. code-block::

   bin/split_data

Data partitions will be generated in the folder 'data/clients'.  
FEDn relies on a configuration file for the client to connect to the server. Create a file called 'client.yaml' with the follwing content:

.. code-block::

   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092

(optional) Use the APIClient to fetch the client configuration and save it to a file:
.. code:: python

   >>> import yaml
   >>> config = client.get_client_config(checksum=True)
   >>> with open("client.yaml", "w") as f:
   >>>    f.write(yaml.dump(config))

Make sure to move the file ``client.yaml`` to the root of the examples/mnist-pytorch folder.
To connect a client that uses the data partition ``data/clients/1/mnist.pt`` and the config file ``client.yaml`` to the network, run the following docker command:

.. code-block::

   docker run \
  -v $PWD/client.yaml:/app/client.yaml \
  -v $PWD/data/clients/1:/var/data \
  -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.pt \
  --network=fedn_default \
  ghcr.io/scaleoutsystems/fedn/fedn:0.8.0-mnist-pytorch run client -in client.yaml --name client1 

Observe the API Server logs and combiner logs, you should see the client connecting.
You are now ready to start training the model. In the python enviroment you installed FEDn:

.. code:: python

   >>> ...
   >>> client.start_session(session_id="test-session", rounds=3)
   # Wait for training to complete, when controller is idle:
   >>> client.get_controller_status()
   # Show model trail:
   >>> client.get_model_trail()
   # Show model performance:
   >>> client.list_validations()

Please see :py:mod:`fedn.network.api` for more details on the APIClient. 

There is also a Jupyter `Notebook <https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb>`_ version of this tutorial including examples of how to fetch and visualize model validations.

To scale up the experiment, refer to the README at 'examples/mnist-pytorch' (or the corresponding Keras version), where we explain how to use docker-compose to automate deployment of several clients.  
