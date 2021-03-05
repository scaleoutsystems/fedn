# Release Notes

## v0.2.0

### What's new?

- Support for PyTorch models
- It's now possible to have examples in external repositories
- Support for models constructed with the Keras Functional API
- Set maximum number of clients in the settings file
- Now possible to upload the compute package through the UI
- Configure round timeout in the UI

```yaml
combiner:
name:
combinerhost:
combinerport: 12080
max_clients: 50
```

- Added several new views in UI including combiner performance and client training time distribution. 
- Updated look-and-feel for UI.
- Added client allocation policy to spread the clients evenly over the combiners
- Use config for s3 model commits instead of a hard-coded bucket name
- Memory management to prevent combiners from going off
- Reducer, client and combiner now have their own separate Dockerfile definitions

### Fixes

- Harmonizes naming of helper classes
- Many bugfixes and stability improvements

### Other

- Lots of product documentation updates
