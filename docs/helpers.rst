.. _helper-label:

Model marshalling
===================================

In federated learning, model updates need to be serialized and deserialized in order to be 
transferred between clients and server/combiner. There is also a need to write and load models 
to/from disk, for example to transiently store updates during training rounds. 
Furthermore, aggregation algorithms need to perform a range of numerical operations on the 
model updates (addition, multiplication, etc). Since different ML frameworks (TF, Torch, etc) 
have different internal ways to represent model parameters, there is a need to inform the 
framework how to handle models of a given type. In Scaleout Edge, this compatibility layer is the 
task of Helpers.

A helper is defined by the interface in :py:mod:`scaleout-util.scaleoututils.helpers.helperbase.HelperBase`. 
By implementing a helper plugin, a developer can extend the framework with support for new ML 
frameworks and numerical operations.  

Scaleout Edge ships with a default helper implementation, ``numpyhelper``. 
This helper relies on the assumption that the model update is made up of parameters 
represented by a list of :py:class:`numpy.ndarray` arrays. Since most ML frameworks have 
good numpy support it should in most cases be sufficient to use this helper. 
Both TF/Keras and PyTorch models can be readily serialized in this way. 

See the Keras and PyTorch quickstart examples and :py:mod:`scaleout-util.scaleoututil.helpers.plugins.numpyhelper` 
for further details.

.. meta::
   :description lang=en:
      In federated learning, model updates need to be serialized and deserialized in order to be transferred between clients and server/combiner.
   :keywords: Federated Learning, Model marshalling, Federated Learning Framework, Federated Learning Platform, FEDn, Scaleout Systems
   
