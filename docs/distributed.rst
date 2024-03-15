Distributed Deployment
===================================

This tutorial outlines the steps for deploying the FEDn framework on a local network, using a workstation as 
the host and different devices as clients. For general steps on how to run FEDn, see one of the quickstart tutorials. 


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


Note the local IP address of your host device; it will be needed later. Follow the standard procedure 
to initiate a FEDn network. Once the network is active, upload your compute package and seed (for comprehensive details, 
see the quickstart tutorials).


Configuring and Attaching Clients
-------------

On your client device, continue with initializing your client. To connect to the host machine we need to ensure we are 
routing the correct DNS to our hosts local IP adress. We can do this using the standard FEDn `client.yaml`:

.. code-block::

   network_id: fedn-network
   discover_host: api-server
   discover_port: 8092


We can then run using docker by adding the hosts to the docker run command:

.. code-block::

   docker run \
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