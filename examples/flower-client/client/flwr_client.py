"""Flower client code using the ClientApp abstraction.

Code adapted from https://github.com/adap/flower/tree/main/examples/app-pytorch.
"""

from flwr.client import ClientApp, NumPyClient
from flwr_task import DEVICE, Net, get_weights, load_data, set_weights, test, train


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, cid) -> None:
        super().__init__()
        print(f"STARTED CLIENT WITH CID {cid}")
        self.net = Net().to(DEVICE)
        self.trainloader, self.testloader = load_data(
            partition_id=int(cid), num_clients=10
        )

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(self.net, self.trainloader, self.testloader, epochs=3)
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient(cid).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
