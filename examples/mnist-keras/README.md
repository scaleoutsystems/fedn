# MNIST (TensorFlow/Keras version)

This is a TF/Keras version of the [Quickstart Tutorial (PyTorch)](https://fedn.readthedocs.io/en/stable/quickstart.html). For more detailed explaination including a Jupyter Notebook with 
examples of API usage for starting and interacting with federated experiments, refer to that tutorial.

## Prerequisites
- [Python >=3.8, <=3.11](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

## Running the example

Clone FEDn and locate into this directory.
```sh
git clone https://github.com/scaleoutsystems/fedn.git
cd fedn/examples/mnist-keras
```

### Preparing the environment, the local data, the compute package and seed model

Build a virtual environment (note that you might need to install the 'venv' package):

#### Ubuntu 

```sh
bin/init_venv.sh
```

#### MacOS with M1 or M2 processors
you need another Tensorflow package, as specified in 'requirements-macos.txt' 

```sh
bin/init_venv_macm1.sh
```

Activate the virtual environment:

```sh
source .mnist-keras/bin/activate
```

Make the compute package (to be uploaded to FEDn):
```sh
tar -czvf package.tgz client
```

Create the seed model (to be uploaded to FEDn):
```sh
python client/entrypoint init_seed
```

Then, to get the data we will use for the clients (MNIST), you can run the following script.
```sh
bin/get_data
```

The next command splits the data in 2 parts for the clients.
```sh
bin/split_data
```
> **Note**: run with `--n_splits=N` to split in *N* parts.


Next, you will upload the compute package and seed model to a FEDn network. Here you have two main options: 
using FEDn Studio (recommended for new users), or a pseudo-local deployment on your own machine.

### If you are using FEDn Studio (recommended):

Follow instructions here to register for Studio and start a project: https://fedn.readthedocs.io/en/stable/studio.html.

In your Studio project: 

- From the "Sessions" menu, upload the compute package and seed model. 
- Register a client and obtain the corresponding 'client.yaml'.  



### Deploy FEDn 
Now we are ready to deploy FEDn and two clients with `docker-compose`.

```
docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up  
```

> **Note**: run with `--scale client=N` to start *N* clients.

### Run federated training 
Refer to the notebook to create your own drivers for seeding the federation and running experiments.

 https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb 

## Clean up
You can clean up by running `docker-compose down`.
