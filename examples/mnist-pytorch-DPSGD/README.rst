FEDn Project: Federated Differential Privacy MNIST (Opacus + PyTorch)
----------------------------------------------------------------------

This example FEDn Project demonstrates how Differential Privacy can be integrated to enhance the confidentiality of client data.
We have expanded our baseline MNIST-PyTorch example by incorporating the Opacus framework, which is specifically designed for PyTorch models. To learn more about differential privacy, read our `blogpost <https://www.scaleoutsystems.com/post/guaranteeing-data-privacy-for-clients-in-federated-machine-learning>`__  about it.



Prerequisites
-------------

-  `Python >=3.9, <=3.12 <https://www.python.org/downloads>`__
-  `A project in FEDn Studio  <https://scaleout.scaleoutsystems.com/signup>`__   


Edit Client-Specific Differential Privacy Parameters 
--------------------------
The **Differential Privacy budget** (``epsilon``, ``delta``), along with other settings, is configurable in the ``client_settings.yaml`` file:

- **epochs**: Number of local epochs per round.
- **epsilon**: Total epsilon budget to spend, determined by the ``global_rounds`` set on the server side.
- **delta**: Total delta budget to spend.
- **max_grad_norm**: Clipping threshold for gradients.
- **global_rounds**: Number of rounds the server will run.
- **hardlimit**:

   - If ``hardlimit`` is set to ``True``, the ``epsilon`` budget will not exceed its specified limit, even if it means skipping updates for some rounds.
   - If ``hardlimit`` is set to ``False``, the expected ``epsilon`` will be approximately equal to its specified value, assuming the server completes the specified ``global_rounds`` of updates.

Creating the compute package and seed model
-------------------------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout.git
   cd fedn/examples/mnist-pytorch-DPSGD

Create the compute package:

.. code-block::

   scaleout package create --path client

This creates a file 'package.tgz' in the project folder.

Next, generate the seed model:

.. code-block::

   scaleout run build --path client

This will create a model file 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  

Running the project on FEDn
----------------------------

To learn how to set up your FEDn Studio project and connect clients, take the quickstart tutorial: https://scaleout.readthedocs.io/en/stable/quickstart.html. 
