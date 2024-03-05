.. _getting_started:

Getting Started
===============

Introduction
------------

Welcome to the documentation for the FEDn Python Package! This guide will walk you through the process of installing the package, setting up your environment, and getting started with using FEDn for your machine learning projects.

Installation
------------

To install the FEDn Python Package, you can use pip:

.. code-block:: bash

   pip install fedn

Setting Up Your Environment
--------------------------

Before you can start using FEDn, you need to set up your initialize the federated network. 
If you are using FEDn Studio, you can skip this step. If running FEDn locally, you need to start the controller and combiner:

.. code-block:: bash

    fedn run

This will start the controller and combiner on your local machine.

Usage
-----

Now that you have installed the FEDn Python Package and set up your environment, you can start using it in your machine learning projects. Here are some examples to get you started:

Listen for model update requests and training a local model:

   .. code-block:: python

      import fedn

      # Set the controller URI
      fedn.set_controller_uri('http://localhost:8092')

      # Use the controller service discovery to find the available combiner (aggregation server)
      combiner = fedn.get_available_combiner()

      # Start listening for incoming training requests
      for task in fedn.listen_for_training_tasks(combiner):

        # Get the global model id 
        global_model_id = task.model_id

        # Get metadata from task, e.g hyperparameters
        params = task.metadata

        # Stage the global model on disk
        fedn.stage_model(global_model_id, path='path/to/models')

        # Or if model can fit into memory
        global_model = fedn.load_model(global_model_id)

        # load dataset from sklearn
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the sklean model with the dataset
        from sklean import linear_model
        model = linear_model.LogisticRegression()

        # Set the hyperparameters and weights from global model
        model.set_params(**params)
        model.coef_ = global_model.coef_

        # Train the model
        model.fit(X_train, y_train)

        # Send the updated model to the combiner
        fedn.send_model_to_combiner(model, combiner)
 
The code snippet will listen for incoming training requests from the combiner. When a request is received, the global model is staged on disk and the metadata is extracted from the task. The global model is then loaded and trained with the local dataset. The updated model is then sent to the combiner.
Observe that this will contunue for ever until the process is killed.

To control the longevity of the process, there are various ways to handle the process. One way is to use the `stop_listening` method to stop the listening process. This requires that the `listen_for_training_tasks` method is called in a separate thread.
You can do usinf the built-in `fedn.start_client` method using a context manager:

    .. code-block:: python
    
        import fedn
    
        # Set the controller URI
        fedn.set_controller_uri('http://localhost:8092')
    
        # Use the controller service discovery to find the available combiner (aggregation server)
        combiner = fedn.get_available_combiner()
    
        # Start listening for incoming training requests
        with fedn.start_client(combiner) as client:
          for task in client.listen_for_training_tasks():
    
            # Do the training and sending of the model as in the previous example
            # ...

This will start the listening process in a separate thread and stop it when the context manager is exited or when `fedn.stop_listening` is called.

Another way is to use the `listen_for_training_tasks` method in a separate thread and use a `threading.Event` to signal the thread to stop listening:

    .. code-block:: python
    
        import fedn
        import threading
    
        # Set the controller URI
        fedn.set_controller_uri('http://localhost:8092')
    
        # Use the controller service discovery to find the available combiner (aggregation server)
        combiner = fedn.get_available_combiner()
    
        # Start listening for incoming training requests
        stop_event = threading.Event()
        thread = threading.Thread(target=listen_for_tasks, args=(combiner, stop_event))
        thread.start()
    
        # Do other stuff
        # ...
    
        # Signal the thread to stop listening
        stop_event.set()
        thread.join()
    
        def listen_for_tasks(combiner, stop_event):
            for task in fedn.listen_for_training_tasks(combiner):
                if stop_event.is_set():
                break
                # Do the training and sending of the model as in the previous example
                # ...



   
Controlling request types
-------------------------
So far we have only listened to training requests. There are also other types of requests that can be listened to. The `listen_for_training_tasks` method is just a wrapper around the `listen_for_tasks` method. The `listen_for_tasks` method can be used to listen to any type of request. The request type is specified as an argument to the method. 
The request types are defined in the `fedn.proto` module. The following request types are available:

- `fedn.proto.TaskType.TRAIN`: A training request, to update a local model with local training data
- `fedn.proto.TaskType.AGGREGATE`: An aggregation request, to aggregate the local models into a global model. Mainly used by the combiner, but can be used to test new aggregation plugins. 
- `fedn.proto.TaskType.VALIDATION`: An validation/evaluation request. Make a local evaluation of the model and send the result to the combiner.
- `fedn.proto.TaskType.INFERENCE`: An inference request. Edge inference, to make a local inference with the model and send the prediction to the combiner. The prediction can be stored either as an object in the object store backend or as a JSON-like object in the task metadata.
  
An example using the `listen_for_tasks` method to listen for validation requests:

    .. code-block:: python
    
        import fedn
        from fedn.proto import TaskType
    
        # Set the controller URI
        fedn.set_controller_uri('http://localhost:8092')
    
        # Use the controller service discovery to find the available combiner (aggregation server)
        combiner = fedn.get_available_combiner()
    
        # Start listening for incoming validation requests
        for task in fedn.listen_for_tasks(combiner, task_type=TaskType.VALIDATION):
    
            # Same procedure as in the previous training example, i.e load the global model, 
            # but instead of training the model, evaluate it and send the result to the combiner
            # ...
            # Set the hyperparameters and weights from global model
            model.set_params(**params)
            model.coef_ = global_model.coef_
    
            # Evaluate the model
            score = model.score(X_test, y_test)
    
            # Send the evaluation result to the combiner
            fedn.send_validation_result_to_combiner(score, combiner, task.task_id, type='infer')


The `type = 'infer'` argument is used to specify the type of validation result. The type can be either `infer` or a specific datatypte. If `type = 'infer'` fedn will try to infer the datatype of the result metric object.

To use INFERENCE requests, the `listen_for_tasks` method can be used to listen for inference requests:

    .. code-block:: python
    
        import fedn
        from fedn.proto import TaskType
    
        # Set the controller URI
        fedn.set_controller_uri('http://localhost:8092')
    
        # Use the controller service discovery to find the available combiner (aggregation server)
        combiner = fedn.get_available_combiner()
    
        # Start listening for incoming inference requests
        for task in fedn.listen_for_tasks(combiner, task_type=TaskType.INFERENCE):
    
            # Same procedure as in the previous training example, i.e load the global model, 
            # but instead of training the model, make an inference and send the result to the combiner
            # ...
            # Set the hyperparameters and weights from global model
            model.set_params(**params)
            model.coef_ = global_model.coef_
    
            # Make an inference
            prediction = model.predict(X_test)
    
            # Send the prediction to the combiner
            fedn.send_inference_result_to_combiner(prediction, combiner, task.task_id, type='infer')


This will listen for inference requests and make an inference with the model and send the result to the combiner. However, in some use-cases a client want to send predictions based on some event. For this purpose, one can use the `send_inference_result_to_combiner` method directly:

    .. code-block:: python
    
        import fedn
    
        # Set the controller URI
        fedn.set_controller_uri('http://localhost:8092')
    
        # Use the controller service discovery to find the available combiner (aggregation server)
        combiner = fedn.get_available_combiner()
    
        # Load the global model. Observe that we assume the client knows the lastest global model id
        # The model can be loaded from disk (already staged) or from the object store
        global_model_id = 'some_global_model_id'
        global_model = fedn.load_model(global_model_id)
    
        # Make an inference
        prediction = global_model.predict(X_test)
    
        # Send the prediction to the combiner
        fedn.send_inference_result_to_combiner(prediction, combiner, task_id='some_task_id', type='infer')

This can be a good option if the client wants to send predictions based on some event, e.g a sensor event based in streaming data.

Conclusion
----------

This guide has walked you through the process of installing the FEDn Python Package, setting up your environment, and getting started with using FEDn for your machine learning projects. 
You should now be able to start using FEDn in your machine learning projects. If you have any questions or need further assistance, please refer to the `FEDn documentation <https://fedn.readthedocs.io>`_ or the `FEDn GitHub repository <

