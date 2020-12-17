# Release Notes

## v0.2.0

### What's new?

- It's now possible to have examples in external repositories
- Support for models constructed with the Keras Functional API
- Set maximum number of clients in the settings file

```yaml
combiner:
name:
combinerhost:
combinerport: 12080
max_clients: 50
```

- Added visualizations on FEDn communication performance to the dashboard
- Added client allocation policy to spread the clients evenly over the combiners
- Use config for s3 model commits instead of a hard-coded bucket name
- Memory management to prevent combiners from going off
- Now possible to upload the compute package through the UI
- Reducer, client and combiner now have their own Dockerfile definitions

### Fixes

- Combiners now handle the case when all clients fail to update a model

### Other

- Lots of product documentation updates
