# MNIST test project
This classsic example of hand-written text recognition is well suited both as a lightweight test when learning FEDn and developing on FEDn in psedo-distributed mode. A normal high-end laptop or a workstation should be able to sustain at least 5 clients. The example is also useful for general scalability tests in fully distributed mode. 

To make testing flexible, clients create their own local dataset upon first training invocation, by sampling from the full dataset in CSV format (can be downloaded from e.g. https://www.kaggle.com/oddrationale/mnist-in-csv, but there are several hosted versions available).  

## Configuring the tests
We have made it possible to configure a couple of settings to vary the conditions for the training. These configurataions are expsosed in the file 'mnist_settings.yaml': 

```yaml 
# Number of training samples used by each client
training_samples: 600
# Number of test samples used by each client (validation)
test_samples: 100
# How much to bias the client data samples towards certain classes (non-IID data partitions)
bias: 0.7
# Parameters for local training
batch_size: 32
epochs: 1
```

## Create the compute package
To train a model in FEDn you provide the client code (in 'client') as a tarball (you set the name of the package in 'settings-reducer.yaml'). For convenience, we ship a pre-made package. Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the code (as a .tar.gz archive) and copy the updated package to 'packages'. From 'test/mnist':

```bash
tar -cf mnist.tar client
gzip mnist.tar
cp mnist.tar.gz packages/
```

## Create the seed model
The baseline CNN is specified in the file 'seed/init_model.py'. This script creates an untrained neural network and serialized that to a file, which is uploaded as the seed model for federated training. For convenience we ship a pregenerated seed model in the 'seed/' directory. If you wish to alter the base model, edit 'init_model.py' and regenerate the seed file:

```bash
python init_model.py 
```

## Start a client
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience. From the root directory of the FEDn repository: 

```bash
sudo docker-compose -f docker-compose.local.yaml up --scale client=2 
```
> The above assumes you are testing againts a local pseudo-distributed FEDn network. Use docker-compose.yaml if you are connecting against a reducer part of a distribured setup.

> This assumes that a FEDn network is running and that the client config and extra_hosts are configured correctly. See the FEDn quick start guide for details.    
