FEDn Project: Importer Client
-----------------------------

This is an example FEDn Project on how to design a client that imports client training code rather than running it in a separate process. 
This enables the user to have access to the grpc channel to send information to thecontroller during training. 

   **Note: We recommend that all new users start by taking the Quickstart Tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html** 

Prerequisites
-------------

-  `Python >=3.9, <=3.12 <https://www.python.org/downloads>`__
-  `A project in FEDn Studio  <https://fedn.scaleoutsystems.com/signup>`__   

Creating the compute package and seed model
-------------------------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/importer-client

Create the compute package:

.. code-block::

   fedn package create --path client

This creates a file 'package.tgz' in the project folder.

Running the project on FEDn
----------------------------

.. code-block::

   fedn client start --importer --init client.yaml


To learn how to set up your FEDn Studio project and connect clients, take the quickstart tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html. 
