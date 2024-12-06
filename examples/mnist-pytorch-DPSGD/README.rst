FEDn Project: Federated Differential Privacy MNIST (Opacus + PyTorch)
----------------------------------------------------------------------

This example FEDn Project demonstrates how Differential Privacy can be integrated to enhance the confidentiality of client data.
We have expanded our baseline MNIST-PyTorch example by incorporating the Opacus framework, which is specifically designed for PyTorch models. If you are interested more about Differential Privacy read our [blogpost](https://www.scaleoutsystems.com/post/guaranteeing-data-privacy-for-clients-in-federated-machine-learning) about it 



Prerequisites
-------------

-  `Python >=3.9, <=3.12 <https://www.python.org/downloads>`__
-  `A project in FEDn Studio  <https://fedn.scaleoutsystems.com/signup>`__   

Edit client specific Differential Privacy parameters 
--------------------------
The **Differential Privacy budget** epsilon, delta is together with other settings, client configurtable in the client_settings.yaml.
- epochs - number of local epochs per round
epsilon - total number of epsilon budget to spend, given global_rounds from the server side.
delta - total number of delta budget to spend.
max_grad_norm - clipping threshold
global_rounds - numbers of rounds the server will run.
hardlimit
- If `hardlimit`  is set to `True`, the `epsilon` will not exceed its specified limit on the expanse that not all rounds model updates will be updates.
- If `hardlimit` is set to `False`, the expected `epsilon` will be around its specified value given the server runs `global_rounds` nr of updates.

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
