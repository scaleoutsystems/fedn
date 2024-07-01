FEDn Project: Keras/Tensorflow (MNIST) 
-------------------------------------------

This is a TF/Keras version of the Quickstart Tutorial (PyTorch) FEDn Project. For a step-by-step guide, refer to that tutorial.

   **Note: We recommend all new users to start by following the Quickstart Tutorial: https://fedn.readthedocs.io/en/latest/quickstart.html**

Prerequisites
-------------------------------------------

-  `Python >=3.8, <=3.12 <https://www.python.org/downloads>`__

Creating the compute package and seed model
-------------------------------------------

Install fedn: 

.. code-block::

   pip install fedn

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

Running the project on FEDn
----------------------------

To set up your FEDn Studio project and connect clients, follow this guide: https://fedn.readthedocs.io/en/latest/studio.html. On the 
step "Upload Files", upload 'package.tgz' and 'seed.npz' created above. 

