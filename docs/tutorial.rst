Tutorial: Compute Package
================================================

This tutorial walks you through the design of a *compute package* for a FEDn client. The compute package is a tar.gz bundle of the code to be executed by each data-provider/client.
You will learn how to design the compute package and how to write the entry points for training and validation. Examples are provided for the Keras and PyTorch frameworks, which can be
found in the `examples <https://github.com/scaleoutsystems/fedn/tree/master/examples>`_.

The compute package
-----------------------------

.. image:: img/ComputePackageOverview.png
   :alt: Compute package overview
   :width: 100%
   :align: center

The *compute package* is a tar.gz bundle of the code to be executed by each data-provider/client. 
This package is uploaded to the *Controller* upon initialization of the FEDN Network (along with the initial model). 
When a client connects to the network, it downloads and unpacks the package locally and are then ready to 
participate in training and/or validation. 

The logic is illustrated in the above figure. When the :py:mod:`fedn.network.clients`  
recieves a model update request from the combiner, it calls upon a Dispatcher that looks up entry point definitions 
in the compute package. These entrypoints define commands executed by the client to update/train or validate a model.

Designing the compute package
------------------------------
We recommend to use the project structure followed by most example `projects <https://github.com/scaleoutsystems/fedn/tree/master/examples>`_.
In the examples we have roughly the following file and folder structure:

| project
| ├── client
| │   ├── entrypoint.py
| │   └── fedn.yaml
| ├── data
| │   └── mnist.npz
| ├── requirements.txt
| └── docker-compose.yml/Dockerfile
| 

The "client" folder is the *compute package* which will become a tar.gz bundle of the code to be executed by 
each data-provider/client. The entry points, mentioned above, are defined in the *fedn.yaml*:

.. code-block:: yaml
    
    entry_points:
        train:
            command: python entrypoint.py <args>
        validate:
            command: python entrypoint.py <args>

The training entry point should be a single-input single-output program, taking as input a model update file 
and writing a model update file (same file format). Staging and upload of these files are handled by the FEDn client. A helper class in the FEDn SDK handles the ML-framework 
specific file serialization and deserialization. The validation entry point acts very similar except we perform validation on the 
*model_in* and outputs a json containing a validation scores (see more below). 

Upon training (model update) requests from the combiner, the client will download the latest (current) global model and *entrypoint.py train* will be executed with this model update as input. 
After training / updating completes, the local client will capture the output file and send back the updated model to the combiner. 
For the local execution this means that the program (in this case entrypoint.py) will be executed as:  

.. code-block:: python

   python entrypoint.py train in_model_path out_model_path <extra-args>

A *entrypoint.py* example can look like this:

.. code-block:: python

    import collections
    import math
    import os

    import docker
    import fire
    import torch

    from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

    HELPER_MODULE = 'pytorchhelper'
    NUM_CLASSES = 10

    def _compile_model():
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

        # Return model
        return Net()


    def _load_data(data_path, is_train=True):
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


    def _save_model(model, out_path):
        """ Save model to disk. 

        :param model: The model to save.
        :type model: torch.nn.Module
        :param out_path: The path to save to.
        :type out_path: str
        """
        weights = model.state_dict()
        weights_np = collections.OrderedDict()
        for w in weights:
            weights_np[w] = weights[w].cpu().detach().numpy()
        helper = get_helper(HELPER_MODULE)
        helper.save(weights, out_path)


    def _load_model(model_path):
        """ Load model from disk.

        param model_path: The path to load from.
        :type model_path: str
        :return: The loaded model.
        :rtype: torch.nn.Module
        """
        helper = get_helper(HELPER_MODULE)
        weights_np = helper.load(model_path)
        weights = collections.OrderedDict()
        for w in weights_np:
            weights[w] = torch.tensor(weights_np[w])
        model = _compile_model()
        model.load_state_dict(weights)
        model.eval()
        return model


    def init_seed(out_path='seed.npz'):
        """ Initialize seed model.

        :param out_path: The path to save the seed model to.
        :type out_path: str
        """
        # Init and save
        model = _compile_model()
        _save_model(model, out_path)


    def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
        """ Train model.

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
        x_train, y_train = _load_data(data_path)

        # Load model
        model = _load_model(in_model_path)

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
            'num_examples': len(x_train),
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr
        }

        # Save JSON metadata file
        save_metadata(metadata, out_model_path)

        # Save model update
        _save_model(model, out_model_path)


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
        x_train, y_train = _load_data(data_path)
        x_test, y_test = _load_data(data_path, is_train=False)

        # Load model
        model = _load_model(in_model_path)

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
            # '_get_data_path': _get_data_path,  # for testing
        })
        


The format of the input and output files (model updates) are dependent on the ML framework used. A helper instance :py:mod:`fedn.utils.plugins.pytorchhelper` is used to handle the serialization and deserialization of the model updates. 
The first function (_compile_model) is used to define the model architecture and creates an initial model (which is then used by _init_seed). The second function (_load_data) is used to read the data (train and test) from disk.  
The third function (_save_model) is used to save the model to disk using the pytorch helper module :py:mod:`fedn.utils.plugins.pytorchhelper`. The fourth function (_load_model) is used to load the model from disk, again
using the pytorch helper module. The fifth function (_init_seed) is used to initialize the seed model. The sixth function (_train) is used to train the model, observe the two first arguments which will be set by the FEDn client. 
The seventh function (_validate) is used to validate the model, again observe the two first arguments which will be set by the FEDn client.

Finally, we use the python package fire to create a command line interface for the entry points. This is not required but convenient.    

For validations it is a requirement that the output is saved in a valid json format: 

.. code-block:: python

   python entrypoint.py validate in_model_path out_json_path <extra-args>
 
In the code example we use the helper function :py:meth:`fedn.utils.helpers.helpers.save_metrics` to save the validation scores as a json file. 

The Dahboard in the FEDn UI will plot any scalar metric in this json file, but you can include any type in the file assuming that it is valid json. These values can then be obtained (by an athorized user) from the MongoDB database or using the :py:mod:`fedn.network.api.client`. 

Packaging for distribution
--------------------------
For the compute package we need to compress the *client* folder as .tar.gz file. E.g. using:

.. code-block:: bash

    tar -czvf package.tgz client


This file can then be uploaded to the FEDn network using the FEDn UI or the :py:mod:`fedn.network.api.client`.


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


Once everything works as expected you can start the federated network, upload the tar.gz compute package and the initial model. 
Finally connect a client to the network:

.. code-block:: bash

    docker run \
    -v $PWD/client.yaml:/app/client.yaml \
    -v $PWD/data/clients/1:/var/data \
    -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.pt \
    --network=fedn_default \
    ghcr.io/scaleoutsystems/fedn/fedn:master-mnist-pytorch run client -in client.yaml --name client1 

The container image "ghcr.io/scaleoutsystems/fedn/fedn:develop-mnist-pytorch" is a pre-built image with the FEDn client and the PyTorch framework installed.

