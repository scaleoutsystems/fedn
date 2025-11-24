Frequently asked questions
==========================

Q: How do I remove/replace the compute package?
-----------------------------------------------

We do not provide an out-of-the box way to clear the compute package for a model that has been intitialized. 
This is a security constraint enforced to not allow for arbitrary code package replacement in an already configured federation. 
However, during development of a new model it will be necessary to reinitialize.  

  1. Set a new compute package using the API: 

    .. code:: python

        >>> from scaleout import APIClient
        >>> client = APIClient(host="<controller-host>", token="<access-token>", secure=True, verify=True)
        >>> client.set_package("package.tgz", helper="numpyhelper")
        >>> client.set_initial_model("seed.npz")      

   2. Restart the clients. 

Q: Can I skip fetching the remote package and instead use a local folder when developing the compute package
------------------------------------------------------------------------------------------------------------

Yes, to facilitate interactive development of the compute package you can start a client that uses a local folder 'client' in your current working directory by: 

.. code-block:: bash

    scaleout client start --local-package


Note that in production federations the remote compute package option should in most cases be disallowed. 

Q: How can I define custom aggregation algorithms?
--------------------------------------------------

Scaleout Edge provides several built-in aggregators, but custom aggregation or
server-side behavior can be implemented through the **server functions**
interface. This allows you to override or extend the Combiner-level logic as
needed.

See :ref:`agg-label` and :ref:`server-functions` for details.


Q: What is needed to include additional ML frameworks in Scaleout Edge?
-------------------------------------------------------------------------------------

You need to make sure that Scaleout Edge knows how to serialize and deserialize the model object. If you can 
serialize to a list of numpy ndarrays in your compute package entrypoint (see the Quickstart Tutorial code), you 
can use the built in "numpyhelper". If this is not possible, you can extend the framework with a custom helper, 
see the section about model marshaling: 

:ref:`helper-label`

Q: How do you approach the question of output privacy? 
----------------------------------------------------------------------------------

We take security in (federated) machine learning seriously. Federated learning is a foundational technology that improves input privacy 
in machine learning by allowing datasets to stay local and private, and not copied to a server. Scaleout Edge is designed to provide an industry grade
implementation of the core communication and aggregation layers of federated learning, as well as configurable modules for traceability, logging
etc, to allow the developer balance between privacy and auditability. With `Scaleout Edge <https://scaleoutsystems.com/framework>`__ we add 
functionality for user authentication, authorization, and federated client identity management. As such, The Scaleout Edge Framework provides
a comprehensive software suite for implementing secure federated learning following industry best-practices.     

Going beyond input privacy, there are several additional considerations relating to output privacy and potential attacks on (federated) machine learning systems. 
For an introduction to the topic, see this blog post: 

- `Output Privacy and Federated Machine Learning <https://www.scaleoutsystems.com/post/output-privacy-and-federated-machine-learning>`__

Striking the appropriate balance between system complexity and security becomes a use-case dependent endeavor, and we are happy to 
support projects with guidance on these matters. For an example, one might consider layering differential privacy on top of the aggregation 
to protect against an honest-but-curious server, at the price of a reduced accuracy for the global model. Depending on the privacy requirements, 
the model type, the amount of data, the number of local updates possible during training etc, this may or may not be necessary. 

We are engaged in several cybersecurity projects focused on federated machine learning, do not hesitate to reach out to discuss further
with the Scaleout team.  

- `LEAKPRO: Leakage Profiling and Risk Oversight for Machine Learning Models <https://www.vinnova.se/en/p/leakpro-leakage-profiling-and-risk-oversight-for-machine-learning-models/>`__
- `Validating a System Development Kit for edge federated learning <https://www.vinnova.se/en/p/validating-a-system-development-kit-for-edge-federated-learning/>`__
- `Trusted Execution Environments for Federated Learning: <https://www.vinnova.se/en/p/trusted-execution-environments-for-federated-learning/>`__
- `Robust IoT Security: Intrusion Detection Leveraging Contributions from Multiple Systems <https://www.vinnova.se/en/p/robust-iot-security-intrusion-detection-leveraging-contributions-from-multiple-systems/>`__

.. meta::
   :description lang=en:
      How do you approach the question of output privacy? We take security in (federated) machine learning seriously.
   :keywords: Federated Learning, FAQ, Federated Learning Framework, Federated Learning Platform, FEDn, Scaleout Systems, Scaleout Edge
   
