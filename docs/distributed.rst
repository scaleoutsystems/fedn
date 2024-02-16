Distributed Deployment
===================================

This tutorial outlines the steps for deploying the FEDn framework on a local network, using a workstation as 
the host and Raspberry Pi 5 devices as clients. We'll demonstrate this setup using the PyTorch MNIST example model 
and compute package.

.. note::
   For deployment over a public network, ensure you have:
   - A public IP address
   - Forwarded the necessary ports
   - Implemented necessary security precautions
   Then, follow these steps but substitute the local IP address with your public IP.
   
Prerequisites
-------------
-  `One host workstation and atleast one client device`
-  `Python 3.8, 3.9 or 3.10 <https://www.python.org/downloads>`__
-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Launch a distributed FEDn Network 
-------------


Note the local IP address of your host device; it will be needed later. Clone the FEDn repository and navigate to the cloned directory.
Start the FEDn network using Docker Compose:

.. code-block::

   docker-compose up 

Install the FEDn SDK
-------------

.. note::
    To programmatically interact with the FEDn network use the APIClient.
    Install the FEDn via pip:

    .. code-block:: bash
       
       $ pip install fedn
       # or from source
       $ cd fedn/fedn
       $ pip install . 


Prepare the compute package and seed for the FEDn network
-------------

From your host device, prepare the compute package and model seed. In this tutorial we will use the 
``examples/mnist-pytorch`` example. 

Locate into ``examples/mnist-pytorch`` folder and initialize a virtual environment.

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

Building a Custom Image for Client Devices (Optional)
-------------

The prebuilt Docker images may not be compatible with Raspberry Pi due to system architecture differences. To 
build a custom Docker image:

#. **Clone the Repository on Your Client Device**: Ensure you're in the root directory of the repository.

#. **Build Your Custom Image**: Substitute <custom image name> with your desired image name:

.. code-block::

   docker build —build-arg REQUIREMENTS=examples/mnist-pytorch/requirements.txt -t <custom image name> .

Configuring and Attaching Clients
-------------

The next step is to configure and attach clients. For this we need to download data and make data partitions: 

Download the data:

.. code-block::

   bin/get_data

Split the data in 2 parts for the clients:

.. code-block::

   bin/split_data

Data partitions will be generated in the folder 'data/clients'.  


FEDn relies on a configuration file for the client to connect to the server. Create a file called 'client.yaml' with 
the follwing content:

.. code-block::

   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092


Make sure to move the file ``client.yaml`` to the root of the examples/mnist-pytorch folder.
To connect a client that uses the data partition ``data/clients/1/mnist.pt`` and the config file ``client.yaml`` to 
the network, run the following docker command (make sure to swap host local ip to the one you noted earlier from your 
host machine and the custom image name):

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   -v $PWD/data/clients/1:/var/data \
   -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.pt \
   —add-host=api-server:<host local ip> \
   —add-host=combiner:<host local ip> \
   <custom image name> run client -in client.yaml --name client1


Start a training session
-------------

On the host device, observe the API Server logs and combiner logs, you should see the client connecting.
You are now ready to start training the model. In the python enviroment you installed FEDn:

.. code:: python

   >>> ...
   >>> client.start_session(session_id="test-session", rounds=3)

Clean up
--------
You can clean up by running 

.. code-block::

   docker-compose down