FEDn Project: Keras/Tensorflow (MNIST) 
-------------------------------------------

This is a TF/Keras version of the PyTorch Quickstart Tutorial. For a step-by-step guide, refer to that tutorial.

   **Note: We recommend all new users to start by following the Quickstart Tutorial: https://fedn.readthedocs.io/en/latest/quickstart.html**

Prerequisites
-------------------------------------------

-  `Python >=3.9, <=3.12 <https://www.python.org/downloads>`__

Creating the compute package and seed model
-------------------------------------------

Install fedn and example specific libraries: 

.. code-block::

   pip install fedn
   pip install -r requirements.txt

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-keras

Create the compute package:

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in a global model trail):

.. code-block::

   fedn run build --path client

This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  

Create distributed datasets:

.. code-block::
   
   python client/data.py

Add datapath:

.. code-block::

   export FEDN_DATA_PATH="data/clients/1/mnist.npz"

Running the project on FEDn
----------------------------



To learn how to set up your FEDn Studio project and connect clients, take the quickstart tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html. 

