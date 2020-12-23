# CASA test project
This classsic example of Human Daily Activity Recognition (HDAR) is well suited both as a lightweight test when learning FEDn and developing on FEDn in psedo-distributed mode. A normal high-end laptop or a workstation should be able to sustain at least 5 clients. The example is also useful for general scalability tests in fully distributed mode. 

### Provide local training and test data
For large data transfer reason we uploaded a data folder in this use case to archive.org.
To test this use-case you need to download prepared data that composed 27 apartments (casa's), each apartment data are distributed over 11 clients,  using this link:
https://archive.org/download/data_20201223/data.zip
- Unzip the file
- Copy the content of the unzipped Archive to the data folder under casa directory

### Configuring the tests
We have made it possible to configure a couple of settings to vary the conditions for the training. These configurations are expsosed in the file 'settings.yaml': 

```yaml 
# Number of test size
nr_examples: 1000
# Percentage of test size
test_size: 0.15
# Flag that identify which value will takes (specific sample size or a percentage over all the data)
# percentage= true will consider test_size otherwise will consider nr_examples
percentage: false
# Parameters for local training
batch_size: 32
epochs: 3
```

### Creating a compute package
To train a model in FEDn you provide the client code (in 'client') as a tarball (you set the name of the package in 'settings-reducer.yaml'). For convenience, we ship a pre-made package. Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the code (as a .tar.gz archive) and copy the updated package to 'packages'. From 'test/casa':

```bash
tar -zcvf client.tar.gz client
cp client.tar.gz packages/
```

## Creating a seed model
The baseline LSTM is specified in the file 'seed/init_model.py'. This script creates an untrained neural network and serialized that to a file, which is uploaded as the seed model for federated training. For convenience we ship a pregenerated seed model in the 'seed/' directory. If you wish to alter the base model, edit 'init_model.py' and regenerate the seed file:



```bash
python init_model.py 
```

## Start the client
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience. First, edit 'fedn-network.yaml' to provide information about the reducer endpoint. Then:

```bash
docker-compose -f docker-compose.dev.yaml up --scale client=2 
```
> Note that this assumes that a FEDn network is running (see separate deployment instructions). The file 'docker-compose.local.yaml' is for testing against a local pseudo-distributed FEDn network. Use 'docker-compose.yaml' if you are connecting against a reducer part of a distributed setup and provide a 'extra_hosts' file.
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience. First, edit 'fedn-network.yaml' to provide information about the reducer endpoint. Then:

The easiest way to distribute data across client is to start this command instead of the previous one 
```bash
docker-compose -f docker-compose.decentralised.yaml up --build
```