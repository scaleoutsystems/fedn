Quick Start
===========

Clone this repository, locate into it and start a pseudo-distributed FEDn network using docker-compose:

.. code-block::

   docker-compose up 



This will start up all neccecary components for a FEDn network, execept for the clients.

.. warning:: 
   The FEDn network is configured to use a local Minio and MongoDB instances for storage. This is not suitable for production, but is fine for testing.

.. note::
    You have the option to programmatically interact with the FEDn network using the Python APIClient, or you can use the Dashboard. In these Note sections we will use the APIClient.
    Install the FEDn via pip:

    .. code-block:: bash
       
       $ pip install fedn
       # or from source
       $ cd fedn
       $ pip install . 

Navigate to http://localhost:8090. You should see the FEDn Dashboard, asking you to upload a compute package. The compute package is a tarball of a project. 
The project in turn implements the entrypoints used by clients to compute model updates and to validate a model.  

Locate into 'examples/mnist-pytorch'.  

Start by initializing a virtual enviroment with all of the required dependencies for this project.

.. code-block:: python

   bin/init_venv.sh

Now create the compute package and an initial model:

.. code-block::

   bin/build.sh

Upload the generated files 'package.tgz' and 'seed.npz' in the FEDn Dashboard.

.. note::
   Instead of uploading in the dashboard do:

   .. code:: python

      >>> from fedn import APIClient
      >>> client = APIClient(host="localhost", port=8092)
      >>> client.set_package("package.tgz", helper="pytorchhelper")
      >>> client.set_initial_model("seed.npz")      

The next step is to configure and attach clients. For this we need to download data and make data partitions: 

Download the data:

.. code-block::

   bin/get_data


Split the data in 2 parts for the clients:

.. code-block::

   bin/split_data

Data partitions will be generated in the folder 'data/clients'.  

Now navigate to http://localhost:8090/network and download the client config file. Place it in the example working directory.  

.. note::
   In the python enviroment you installed FEDn:

   .. code:: python

      >>> ...
      >>> client.get_client_config("client.yaml", checksum=True)

To connect a client that uses the data partition 'data/clients/1/mnist.pt': 

.. code-block::

   docker run \
  -v $PWD/client.yaml:/app/client.yaml \
  -v $PWD/data/clients/1:/var/data \
  -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.pt \
  --network=fedn_default \
  ghcr.io/scaleoutsystems/fedn/fedn:develop-mnist-pytorch run client -in client.yaml --name client1 

.. note::
   If you are using the APIClient you must also start the training client via "docker run" command as above.   

You are now ready to start training the model at http://localhost:8090/control.

.. note::
   In the python enviroment you installed FEDn you can start training via:

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

To scale up the experiment, refer to the README at 'examples/mnist-pytorch' (or the corresponding Keras version), where we explain how to use docker-compose to automate deployment of several clients.  
