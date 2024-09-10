FEDn Project: Federated Differential Privacy MNIST (Opacus + PyTorch)
-----------------------------

This example FEDn Project demonstrates how Differential Privacy can be integrated to enhance the confidentiality of client data.
We have expanded our baseline MNIST-PyTorch example by incorporating the Opacus framework, which is specifically designed for PyTorch models.



Prerequisites
-------------

-  `Python >=3.8, <=3.12 <https://www.python.org/downloads>`__
-  `A project in FEDn Studio  <https://fedn.scaleoutsystems.com/signup>`__   

Edit Differential Privacy budget
--------------------------
- The **Differential Privacy budget** (`FINAL_EPSILON`, `DELTA`) is configured in the `compute` package at `client/train.py` (lines 35 and 39).
- If `HARDLIMIT` (line 40) is set to `True`, the `FINAL_EPSILON` will not exceed its specified limit.
- If `HARDLIMIT` is set to `False`, the expected `FINAL_EPSILON` will be around its specified value given the server runs `ROUNDS` variable (line 36).

Creating the compute package and seed model
-------------------------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/mnist-pytorch-DPSGD

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
