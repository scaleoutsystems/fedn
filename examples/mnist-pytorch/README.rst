FEDn Project: MNIST (PyTorch)
-----------------------------

This is an example FEDn Project based on the classic hand-written text recognition dataset MNIST. 
The example is intented as a minimalistic quickstart to learn how to use FEDn.

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
   cd fedn/examples/mnist-pytorch

Create the compute package:

.. code-block::

   fedn package create --path client

This creates a file 'package.tgz' in the project folder.

Next, generate the seed model:

.. code-block::

   fedn run build --path client

This will create a model file 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  

Running the project on FEDn
----------------------------

To learn how to set up your FEDn Studio project and connect clients, take the quickstart tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html. 
