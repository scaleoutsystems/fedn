# MNIST test project
This classic example of hand-written text recognition is well suited both as a lightweight test when learning FEDn and developing on FEDn in psedo-distributed mode. A normal high-end laptop or a workstation should be able to sustain at least 5 clients. The example is also useful for general scalability tests in fully distributed mode. 

> Note that this file descibes how to configure an alliance for a particular FedML-model and how to attach clients. We here assume that a FEDn network is aleady up and running. If this is not the case, start here: https://github.com/scaleoutsystems/fedn/blob/master/README.md

## Setting up a client

### Provide local training and test data
This example is provided with the mnist dataset from https://s3.amazonaws.com/img-datasets/mnist.npz in 'data/mnist.npz'.
To make testing flexible, each client subsamples from this dataset upon first invokation of a training request, then cache this subsampled data for use for the remaining lifetime of the client. It is thus normal that the first training round takes a bit longer than subssequent ones.  

### Create and upload a compute package
To train a model in FEDn you provide the client code (in 'client') as a tarball. For convenience, we ship a pre-made package. Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the code (as a .tar.gz archive) and copy the updated package to 'packages'. From 'test/mnist-keras':

```bash
tar -cf mnist.tar client
gzip mnist.tar
cp mnist.tar.gz packages/
```

Navigate to 'https://localhost:8090/start' and follow the link to 'context' to upload the compute package. 
 
## Creating a seed model
The baseline CNN is specified in the file 'client/init_model.py'. This script creates an untrained neural network and serializes that to a file, which is uploaded as the seed model for federated training. For convenience we ship a pregenerated seed model in the 'seed/' directory. If you wish to alter the base model, edit 'init_model.py' and regenerate the seed file (install dependencies as needed):

```bash
python init_model.py 
```

Navigate to 'localhost:8090/history' to upload the seed model. 

## Start the client
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience:

```bash
docker-compose up --scale client=2 
```

> Note that this assumes that a FEDn network is running in pseudo-distributed mode (see separate deployment instructions) and uses the default service names. If you are connecting to a reducer part of a distributed setup, first, edit 'fedn-network.yaml' to provide IP address to the reducer. Also provide an 'extra_hosts.yaml' file with combiner:host mappings (edit the file according to your network)

```bash
docker-compose -f docker-compose.yaml -f extra-hosts.yaml up 
```

When clients are running, navigate to 'localhost:8090/start' to start the training. 

### Configuring the tests
We have made it possible to configure a couple of settings to vary the conditions for the training. These configurations are expsosed in the file 'settings.yaml': 

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
