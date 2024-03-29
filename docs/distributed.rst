Distributed Deployment
===================================

This tutorial outlines the steps for deploying the FEDn framework over a **local network**, using a workstation as 
the host and different devices as clients. For general steps on how to run FEDn, see one of the quickstart tutorials. 


.. note::
   For a secure and production-grade deployment solution over **public networks**, explore the FEDn Studio service at 
   **studio.scaleoutsystems.com**. 
   
   Alternatively follow this tutorial substituting the hosts local IP with your public IP, open the neccesary 
   ports (see which ports are used in docker-compose.yaml), and ensure you have taken additional neccesary security 
   precautions.
   
Prerequisites
-------------
-  `One host workstation and atleast one client device`
-  `Python 3.8, 3.9 or 3.10 <https://www.python.org/downloads>`__
-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__

Launch a distributed FEDn Network 
-------------


Start by noting your host's local IP address, used within your network. Discover it by running ifconfig on UNIX or 
ipconfig on Windows, typically listed under inet for Unix and IPv4 for Windows.

Continue with following the standard procedure to initiate a FEDn network, for example using by docker-compose. 
Once the network is active, upload your compute package and seed (for comprehensive details, see the quickstart tutorials).


Configuring and Attaching Clients
-------------

On your client device, continue with initializing your client. To connect to the host machine we need to ensure we are 
routing the correct DNS to our hosts local IP address. We can do this using the standard FEDn `client.yaml`:

.. code-block::

   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092


We can then run using docker by adding the hosts in the docker run command:

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


Start a training session
-------------

After connecting with your clients, you are ready to start training sessions from the host machine.