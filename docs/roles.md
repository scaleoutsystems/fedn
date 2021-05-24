# Key roles in the lifecycle of a federated model: 

Three distinct technical roles can be distinguished in the life cycle of a federated model, the network manager, the model initiator and the data provider. The [getting started guide](README.md), showcases all these roles for a pseudo-ditributed deployment.  

## Network manager - the person / organization responsible for running the FEDn network. 

- Runs the FEDn controller and auxiliary services (databases etc.)
- Plans the deployment strategy and topology of the network and deploys and manages combiners. 
- Responsible for end-to-end enterprise security (configures networks, certificates etc)
- Typically a data engineer / DevOps engineer

Please refer to the getting started guide, and the [deployment guidelines](deployment.md)  

## Model initiator - Develops the code associated with the FedML model. 

- Develops the base model (the actual machine learning model). 
- Develops the code to be executed by clients during model updates and validations.
- Describes data formats and pre-processing requirements.
- Configures shared global validation pipelines (optionally)
- Packages the model in a compute package and initializes the network.
- Provides execution environment definitions (typically as Docker-templates) 
- Creates the seed model and uploads that to the network.  
- Configures FEDn [control strategies](control.md) (algorithm)

This [tutorial](tutorial.md) details the model inititor tasks. 
 
## Data provider: Attaches to the network and provides local data. 

- Prepares local data according to instructions from model initiator. 
- Attaches clients to the network. 

[STACKn](https://github.com/scaleoutsystems/stackn) provides additional tools to support the *network manager* and *model initiator* roles. 

