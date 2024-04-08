.. _projects-label:

FEDn Projects
================================================

A FEDn project is a convention for packaging/wrapping machine learning code to be used for federated learning with FEDn. At the core, 
a project is a directory of files (often as a Git repository), containing your machine learning code, FEDn entrypoints, and a specification 
of the runtime environment (python environment or a Docker image). The FEDn API and command-line tools helps a user automate deployment of
a project that follows the conventions. 

Overview
------------------------------

We recommend that projects have roughly the following folder and file structure:

| project
| ├── client
| │   ├── fedn.yaml
| │   ├── python_env.yaml
| │   ├── data.py
| │   ├── model.py
| │   ├── train.py
| │   └── validate.py
| ├── data
| │   └── mnist.npz
| ├── README.md
| ├── scripts / notebooks
| └── Dockerfile / docker-compose.yaml
| 

The "client" folder is referred to as the *compute package*. When deploying the project to FEDn, this folder will be compressed as a .tgz bundle and uploaded to the FEDn controller. 
FEDn will automatically distribute this bundle to each connected client/data provider. Upon recipt of the bundle, the client will unpack it and stage it locally, then initialize a Dispatcher
ready to execute code for computing model updates (local training) and (optionally) validating models. The Dispatcher will look to the FEDn Project File 'fedn.yaml' for entrypoints to execute.

.. image:: img/ComputePackageOverview.png
   :alt: Compute package overview
   :width: 100%
   :align: center

The above figure provides a logical view of how FEDn uses the compute package (client folder). When the :py:mod:`fedn.network.clients`  
recieves a model update request, it calls upon a Dispatcher that looks up entry point definitions 
in the compute package. These entrypoints define commands executed by the client to update/train or validate a model.   

FEDn Project File (fedn.yaml)
------------------------------

FEDn uses on a project file named 'fedn.yaml' to specify which entrypoints to execute when the client recieves a training or validation request, and 
what environment to execute those entrypoints in. 

.. code-block:: yaml

    python_env: python_env.yaml

    entry_points:
        startup:
            command: python data.py
        train:
            command: python train.py
        validate:
            command: python validate.py


Environment
---------------
 
The software environment to be used to exectute the entry points. This should specify all client side dependencies of the project. 
FEDn currently supports Virtualenv environments, with packages on PyPI. When a project specifies a **python_env**, the FEDn 
client will create an isolated virtual environment and install the project dependencies into it before strating up the client.  


Entry Points
------------------------------

There are up to three Entry Points to be specified.


Startup Entrypoint (startup, optional): 

This entrypoint is called **once** immediately after the client starts up. It can be used to do runtime configurations of the local execution environment. 
For example, in the quickstart tutorial example, it is used to download the MNIST dataset and create partitions. This is a convenience useful for 
automation of experiments. 

Training Entrypoint (train,  mandatory): 

This entrypoint is invoked every time the client recieves a new model update request. The training entry point must be a single-input single-output (SISO) program. It will be invoked by FEDn as such: 

.. code-block:: python

    python train.py model_in model_out

where in_model_path is the path to a model update file, and out_model_path is the file to which the new model update will be written. 

Download and upload of these files are handled automatically by the FEDn client. 

Validation Entrypoint (validate, optional): 

The validation entry point acts very similar to the traning entrypoint. It should read a model update from file, validate it (in any way suitable to the user), and write  a **json file** containing validation data:

.. code-block:: python

    python validate.py model_in validations.json


.. code-block:: python

    import collections
    import math
    import os

    import docker
    import fire
    import torch

    from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

    HELPER_MODULE = 'numpyhelper'
    helper = get_helper(HELPER_MODULE)

    NUM_CLASSES = 10


    def _get_data_path():
        """ For test automation using docker-compose. """
        # Figure out FEDn client number from container name
        client = docker.from_env()
        container = client.containers.get(os.environ['HOSTNAME'])
        number = container.name[-1]

        # Return data path
        return f"/var/data/clients/{number}/mnist.pt"


    def compile_model():
        """ Compile the pytorch model.

        :return: The compiled model.
        :rtype: torch.nn.Module
        """
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = torch.nn.Linear(784, 64)
                self.fc2 = torch.nn.Linear(64, 32)
                self.fc3 = torch.nn.Linear(32, 10)

            def forward(self, x):
                x = torch.nn.functional.relu(self.fc1(x.reshape(x.size(0), 784)))
                x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
                x = torch.nn.functional.relu(self.fc2(x))
                x = torch.nn.functional.log_softmax(self.fc3(x), dim=1)
                return x

        return Net()


    def load_data(data_path, is_train=True):
        """ Load data from disk.

        :param data_path: Path to data file.
        :type data_path: str
        :param is_train: Whether to load training or test data.
        :type is_train: bool
        :return: Tuple of data and labels.
        :rtype: tuple
        """
        if data_path is None:
            data = torch.load(_get_data_path())
        else:
            data = torch.load(data_path)

        if is_train:
            X = data['x_train']
            y = data['y_train']
        else:
            X = data['x_test']
            y = data['y_test']

        # Normalize
        X = X / 255

        return X, y


    def save_parameters(model, out_path):
        """ Save model paramters to file.

        :param model: The model to serialize.
        :type model: torch.nn.Module
        :param out_path: The path to save to.
        :type out_path: str
        """
        parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
        helper.save(parameters_np, out_path)


    def load_parameters(model_path):
        """ Load model parameters from file and populate model.

        param model_path: The path to load from.
        :type model_path: str
        :return: The loaded model.
        :rtype: torch.nn.Module
        """
        model = compile_model()
        parameters_np = helper.load(model_path)

        params_dict = zip(model.state_dict().keys(), parameters_np)
        state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model


    def init_seed(out_path='seed.npz'):
        """ Initialize seed model and save it to file.

        :param out_path: The path to save the seed model to.
        :type out_path: str
        """
        # Init and save
        model = compile_model()
        save_parameters(model, out_path)


    def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
        """ Complete a model update.

        Load model paramters from in_model_path (managed by the FEDn client),
        perform a model update, and write updated paramters
        to out_model_path (picked up by the FEDn client).

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param out_model_path: The path to save the output model to.
        :type out_model_path: str
        :param data_path: The path to the data file.
        :type data_path: str
        :param batch_size: The batch size to use.
        :type batch_size: int
        :param epochs: The number of epochs to train.
        :type epochs: int
        :param lr: The learning rate to use.
        :type lr: float
        """
        # Load data
        x_train, y_train = load_data(data_path)

        # Load parmeters and initialize model
        model = load_parameters(in_model_path)

        # Train
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        n_batches = int(math.ceil(len(x_train) / batch_size))
        criterion = torch.nn.NLLLoss()
        for e in range(epochs):  # epoch loop
            for b in range(n_batches):  # batch loop
                # Retrieve current batch
                batch_x = x_train[b * batch_size:(b + 1) * batch_size]
                batch_y = y_train[b * batch_size:(b + 1) * batch_size]
                # Train on batch
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                # Log
                if b % 100 == 0:
                    print(
                        f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            'num_examples': len(x_train),
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr
        }

        # Save JSON metadata file (mandatory)
        save_metadata(metadata, out_model_path)

        # Save model update (mandatory)
        save_parameters(model, out_model_path)


    def validate(in_model_path, out_json_path, data_path=None):
        """ Validate model.

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param out_json_path: The path to save the output JSON to.
        :type out_json_path: str
        :param data_path: The path to the data file.
        :type data_path: str
        """
        # Load data
        x_train, y_train = load_data(data_path)
        x_test, y_test = load_data(data_path, is_train=False)

        # Load model
        model = load_parameters(in_model_path)
        model.eval()

        # Evaluate
        criterion = torch.nn.NLLLoss()
        with torch.no_grad():
            train_out = model(x_train)
            training_loss = criterion(train_out, y_train)
            training_accuracy = torch.sum(torch.argmax(
                train_out, dim=1) == y_train) / len(train_out)
            test_out = model(x_test)
            test_loss = criterion(test_out, y_test)
            test_accuracy = torch.sum(torch.argmax(
                test_out, dim=1) == y_test) / len(test_out)

        # JSON schema
        report = {
            "training_loss": training_loss.item(),
            "training_accuracy": training_accuracy.item(),
            "test_loss": test_loss.item(),
            "test_accuracy": test_accuracy.item(),
        }

        # Save JSON
        save_metrics(report, out_json_path)


    if __name__ == '__main__':
        fire.Fire({
            'init_seed': init_seed,
            'train': train,
            'validate': validate,
        })
        


The format of the input and output files (model updates) are using numpy ndarrays. A helper instance :py:mod:`fedn.utils.helpers.plugins.numpyhelper` is used to handle the serialization and deserialization of the model updates. 
The first function (_compile_model) is used to define the model architecture and creates an initial model (which is then used by _init_seed). The second function (_load_data) is used to read the data (train and test) from disk.  
The third function (_save_model) is used to save the model to disk using the numpy helper module :py:mod:`fedn.utils.helpers.plugins.numpyhelper`. The fourth function (_load_model) is used to load the model from disk, again
using the pytorch helper module. The fifth function (_init_seed) is used to initialize the seed model. The sixth function (_train) is used to train the model, observe the two first arguments which will be set by the FEDn client. 
The seventh function (_validate) is used to validate the model, again observe the two first arguments which will be set by the FEDn client.

Finally, we use the python package fire to create a command line interface for the entry points. This is not required but convenient.    

For validations it is a requirement that the output is saved in a valid json format: 

.. code-block:: python

   python entrypoint.py validate in_model_path out_json_path <extra-args>
 
In the code example we use the helper function :py:meth:`fedn.utils.helpers.helpers.save_metrics` to save the validation scores as a json file. 

These values can then be obtained (by an athorized user) from the MongoDB database or using the :py:meth:`fedn.network.api.client.APIClient.list_validations`. 

Packaging for distribution
--------------------------
For the compute package we need to compress the *client* folder as .tgz file. E.g. using:

.. code-block:: bash

    tar -czvf package.tgz client


This file can then be uploaded to the FEDn network using the :py:meth:`fedn.network.api.client.APIClient.set_package`.


More on local data access 
-------------------------

There are many possible ways to interact with the local dataset. In principle, the only requirement is that the train and validate endpoints are able to correctly 
read and use the data. In practice, it is then necessary to make some assumption on the local environemnt when writing entrypoint.py. This is best explained 
by looking at the code above. Here we assume that the dataset is present in a file called "mnist.npz" in a folder "data" one level up in the file hierarchy relative to 
the exection of entrypoint.py. Then, independent on the preferred way to run the client (native, Docker, K8s etc) this structure needs to be maintained for this particular 
compute package. Note however, that there are many ways to accompish this on a local operational level.

Running the client
------------------
We recommend you to test your code before running the client. For example, you can simply test *train* and *validate* by:

.. code-block:: bash

    python entrypoint.py train ../seed.npz ../model_update.npz --data_path ../data/mnist.npz
    python entrypoint.py validate ../model_update.npz ../validation.json --data_path ../data/mnist.npz


Once everything works as expected you can start the federated network, upload the .tgz compute package and the initial model (use :py:meth:`fedn.network.api.client.APIClient.set_initial_model` for uploading an initial model). 
Finally connect a client to the network:

.. code-block:: bash

    docker run \
    -v $PWD/client.yaml:/app/client.yaml \
    -v $PWD/data/clients/1:/var/data \
    -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.pt \
    --network=fedn_default \
    ghcr.io/scaleoutsystems/fedn/fedn:0.8.0-mnist-pytorch run client -in client.yaml --name client1 

The container image "ghcr.io/scaleoutsystems/fedn/fedn:0.8.0-mnist-pytorch" is a pre-built image with the FEDn client and the PyTorch framework installed.

