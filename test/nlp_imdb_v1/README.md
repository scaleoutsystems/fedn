# IMDB test project
This classic example of sentiment analysis is well suited both as a lightweight test when learning FEDn and developing on FEDn in psedo-distributed mode. A normal high-end laptop or a workstation should be able to sustain at least 5 clients. The example is also useful for general scalability tests in fully distributed mode. 

### Provide local training and test data
For large data transfer reason we uploaded a data folder to archive.org.
To test this use-case you need to download data from this link
https://archive.org/download/data_20201217/data.zip, and decompress the file
and then copy the content to the data folder in nlp_imdb

### Configuring the tests
We have made it possible to configure a couple of settings to vary the conditions for the training. These configurations are expsosed in the file 'settings.yaml': 

```yaml 
# Parameters for local training
test_size: 0.25
batch_size: 32
epochs: 1
```

### Creating a compute package
To train a model in FEDn you provide the client code (in 'client') as a tarball (you set the name of the package in 'settings-reducer.yaml'). For convenience, we ship a pre-made package. Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the code (as a .tar.gz archive) and copy the updated package to 'packages'. From 'test/nlp_imdb_senti':

```bash
tar -cf nlp_imdb_senti.tar client
gzip nlp_imdb_senti.tar
cp nlp_imdb_senti.tar.gz packages/
```

## Creating a seed model
The baseline CNN-LSTM is specified in the file 'seed/init_model.py'. This script creates an untrained neural network and serialized that to a file, which is uploaded as the seed model for federated training. For convenience we ship a pregenerated seed models(CNN-LSTM : 879fa112-c861-4cb1-a25d-775153e5b550.h5) in the 'seed/' directory. If you wish to alter the base model, edit 'init_model.py' and regenerate the seed file:
This example assumes that training and test data is available as 'test/nlp_imdb/data/train.csv'. Data can be downloaded from e.g. https://archive.org/download/data_20201119/data.zip, but there are several hosted versions available. To make testing flexible, each client sub-samples from this dataset upon first invocation of a training request, then cache this sub-sampled data for use for the remaining lifetime of the client. The subsample size is configured as described in the next section. 

```bash
python init_model.py 
```

## Start the client
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience. First, edit 'fedn-network.yaml' to provide information about the reducer endpoint. Then:

```bash
docker-compose -f docker-compose.local.yaml up --scale client=2 
```
> Note that this assumes that a FEDn network is running (see separate deployment instructions). The file 'docker-compose.local.yaml' is for testing against a local pseudo-distributed FEDn network. Use 'docker-compose.yaml' if you are connecting against a reducer part of a distributed setup and provide a 'extra_hosts' file.
The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience. First, edit 'fedn-network.yaml' to provide information about the reducer endpoint. Then:

The easiest way to distribute data across client is to start this command instead of the previous one 
```bash
docker-compose -f docker-compose.decentralised.yaml up --build
```