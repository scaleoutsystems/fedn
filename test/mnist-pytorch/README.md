# MNIST test project (PyTorch version)
This classic example of hand-written text recognition is well suited both as a lightweight test when learning FEDn and developing on FEDn in psedo-distributed mode. A normal high-end laptop or a workstation should be able to sustain at least 5 clients. The example is also useful for general scalability tests in fully distributed mode. 

## Setting up a client

> Note that this assumes that a FEDn network is up and running with the "pytorch" helper, which is identified in "config/settings-reducer.yaml" (see separate deployment instructions). If you are connecting against a reducer part of a distributed setup and provide a 'extra_hosts' file.

### Provide local training and test data
This example is provided with the mnist dataset from https://s3.amazonaws.com/img-datasets/mnist.npz in 'data/mnist.npz'. 
To make testing flexible, each client subsamples from this dataset upon first invokation of a training request, then cache this subsampled data for use for the remaining lifetime of the client. It is thus normal that the first training round takes a bit longer than subssequent ones.

## Configure reducer

Navigate to `https://localhost:8090/` to start configuring the reducer.

### Step 1 Use the provided compute package
Following the guide to configure the reducer you will be prompted to upload a compute package.
For this example please select `PyTorch` as helper and select the file in the folder `package/` as provided.

### _Step 1 - alternative Creating a compute package (optional)_
To train a model in FEDn you provide the client code (in 'client') as a tarball (you set the name of the package in 'settings-reducer.yaml'). For convenience, we ship a pre-made package. Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the code (as a .tar.gz archive), clear the database, restart the reducer and re-upload the package. From 'test/mnist':

```bash
tar -cf mnist.tar client
gzip mnist.tar
cp mnist.tar.gz packages/
```

### Step 2 - Use the provided initial model 
Existing in `initial_model` folder, select the file from the reducer configuration guide and upload by following the prompted messages.

### _Step 2 - alternative Creating a initial model (optional)_
The baseline CNN is specified in the file 'client/init_model.py'. This script creates an untrained neural network and serialized that to a file, which is uploaded as the seed model for federated training. For convenience we ship a pregenerated seed model in the 'seed/' directory. If you wish to alter the base model, edit 'init_model.py' and regenerate the initial model file:

```bash
python init_model.py 
```

### Step 3 - Start the client
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience:

```bash
docker-compose up -f ../../config/private-network.yaml --scale client=2
```
> Note that this assumes that a FEDn network is running in pseudo-distributed mode (see separate deployment instructions) and uses the default service names. If you are connecting to a reducer part of a distributed setup, first, edit 'fedn-network.yaml' to provide information about the reducer endpoint. Then run following command in project directory: provide a 'extra_hosts' file with combiner:host mappings (edit the file according to your network)

```bash
docker-compose -f docker-compose.yaml -f ../../config/private-network.yaml -f extra-hosts.yaml up 
```

When clients are running, navigate to 'localhost:8090/start' to start the training.
### Step 4 - Start the Federation
Return to the reducer and configure a desired run configuration under `https://localhost:8090/control`

Start and watch dashboard and network to observe progress as the example federation iterates on federated the model.

## Configuring the example

### Configuring the tests (optional)
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
