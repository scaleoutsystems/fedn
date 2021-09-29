Release Notes
=============

v0.2.5
------

What's new?
...........

 - The examples previously residing in 'test' have been refactored into a separate repository: https://github.com/scaleoutsystems/examples 
 - Docker-compose templates for Minio upgraded to support latest version 

Other
.....

 - Documentation updated
 - Introduce Discord community server
 

v0.2.4
------

What's new?
...........

- Introduced a new events view.
- Introduced a new view for viewing network layout, (reducer, combiner and clients hierarchy)
- Introduced a new setup guide-phase to ensure prereqs like package and model are set before starting execution.
- Introduced a better form for parameter selection on run configuration.
- Introduced async dispatching of run configurations.
- Introduced async update refresh of several important fields for user convenincence like status, events, network hierarchy etc.
- Introduced a new download-client-config function to allow for faster and more convenient client configuration.
  (Just download config and point your local client and whoallah! You are online in this federation.)
  
Other
.....

- Fixed logic bugs related to framework persistance.
- Fixed a logic bug causing clients to get assigned prior to compute package assignment (and hence will not account for assignment policy).
- Fixed a logic bug if reducer is resumed from previous state (to ensure) that the right compute package is selected.
- Update dependency versions.

v0.2.3
------

What's new?
...........

- Support for latest Minio
- Improvements i UI - now not possible to submit jobs is in monitoring state.
- Improvement of Docker image hierarchy.

Other
.....

- Docs updates 
- Several bugfixes and security patches.


v0.2.2
------

What's new?
...........

- The MNIST examples (Keras and PyTorch) have been updated so that they now bundle the example data in .npz format.

Other
.....

- Docs updates 

v0.2.1
------

What's new?
...........

- It is now possible to choose which validation metrics to plot in the Dashboard

Fixes
.....

- Plots backed by no current data is no longer shown as empty plots. 

Other
.....

- Docs updates 

v0.2.0
------

What's new?
...........

- Support for PyTorch models
- It's now possible to have examples in external repositories
- Support for models constructed with the Keras Functional API
- Set maximum number of clients in the settings file
- Now possible to upload the compute package through the UI
- Configure round timeout in the UI

.. code-block:: yaml

    combiner:
    name:
    combinerhost:
    combinerport: 12080
    max_clients: 50


- Added several new views in UI including combiner performance and client training time distribution. 
- Updated look-and-feel for UI.
- Added client allocation policy to spread the clients evenly over the combiners
- Use config for s3 model commits instead of a hard-coded bucket name
- Memory management to prevent combiners from going off
- Reducer, client and combiner now have their own separate Dockerfile definitions

Fixes
.....

- Harmonizes naming of helper classes
- Many bugfixes and stability improvements

Other
.....

- Lots of product documentation updates
