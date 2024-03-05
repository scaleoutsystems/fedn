Frequently asked questions
==========================

Q: How do I remove/replace the compute package?
-----------------------------------------------

We do not provide an out-of-the box way to clear the compute package for a model that has been intitialized. 
This is a security constraint enforced to not allow for arbitrary code package replacement in an already configured federation. 
However, during development of a new model it will be necessary to reinitialize.  

  1. Set a new compute package using the API: 

    .. code:: python

        >>> from fedn import APIClient
        >>> client = APIClient(host="localhost", port=8092)
        >>> client.set_package("package.tgz", helper="numpyhelper")
        >>> client.set_initial_model("seed.npz")      

   2. Restart the clients. 

Q: Can I skip fetching the remote package and instead use a local folder when developing the compute package
------------------------------------------------------------------------------------------------------------

Yes, to facilitate interactive development of the compute package you can start a client that uses a local folder 'client' in your current working directory by: 

.. code-block:: bash

    fedn run client --remote=False -in client.yaml 


Note that in production federations this options should in most cases be disallowed. 

Q: How can other aggregation algorithms can be defined?
-------------------------------------------------------

There is a plugin interface for extending the framework with new aggregators. See 

:ref:`agg-label`


Q: What is needed to include other ML frameworks in FEDn like sklearn, xgboost, etc.?
-------------------------------------------------------------------------------------

You need to make sure that FEDn knows how to serialize and deseralize the model object into paramters. If you can 
serialize to a list of numpy ndarrays in your compute package entrypoint (see the Quickstart Tutorial code), you 
can use the built in "numpyhelper". If this is not possible, you can extend the framework with a custom helper, 
see the section about model marshaling: 

:ref:`helper-label`

Q: Can I start a client listening only to training requests or only on validation requests?:
--------------------------------------------------------------------------------------------

Yes! You can toggle which message streams a client subscibes to when starting the client. For example, to start a pure validation client: 

.. code-block:: bash

    fedn run client --trainer=False -in client.yaml 


Q: How do you approach the question of output privacy? 
----------------------------------------------------------------------------------

We take security in (federated) machine learning very seriously. Federated learning is a foundational technology that impoves input privacy 
in machine learning by allowing datasets to stay local and private, and not copied to a server. FEDn is designed to provide an industry grade
implementation of the core communication and aggregration layers of federated learning, as well as configurable modules for traceability, logging
etc, to allow the developer balance between privacy and auditability. With `FEDn Studio <https://scaleoutsystems.com/framework>`__ we add 
functionality for user authentication, authorization, and federated client identity management. As such, The FEDn Framework provides
a comprehensive software suite for implemeting secure federated learning following industry best-practices.     

Going beyond input privacy, there are several additional considerations relating to output privacy and potential attacks on (federated) machine learning systems. For an
introduction to the topic, see this blog post: 

- `Output Privacy and Federated Machine Learning <https://www.scaleoutsystems.com/post/output-privacy-and-federated-machine-learning>`__

Striking the appropriate balance between system complexity and secturity becomes a use-case dependent endeavor, and we are happy to 
engage in detailed conversations about this. As an example, one might consider layering differential privacy on top of the aggregation 
to protect against a honest-but-curious server, at the price of a loss of accuracy for the global model. Depending on the privacy requirements, 
the model type, the amount of data, the number of local updates possible during training etc, this may or may not be necessary. 

We are engaged in several cybersecurity projects focused on federated machine learning, do not hesitate to reach out to discuss further
with the Scaleout team.  

- `LEAKPRO: Leakage Profiling and Risk Oversight for Machine Learning Models <https://www.vinnova.se/en/p/leakpro-leakage-profiling-and-risk-oversight-for-machine-learning-models/>`__
- `Validating a System Development Kit for edge federated learning <https://www.vinnova.se/en/p/validating-a-system-development-kit-for-edge-federated-learning/>`__
- `Truseted Execution Environments for Federated Learning: <https://www.vinnova.se/en/p/trusted-execution-environments-for-federated-learning/>`__
- `Robust IoT Security: Intrusion Detection Leveraging Contributions from Multiple Systems <https://www.vinnova.se/en/p/robust-iot-security-intrusion-detection-leveraging-contributions-from-multiple-systems/>`__