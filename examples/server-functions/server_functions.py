from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

# See allowed_imports for what packages you can use in this class.


class ServerFunctions(ServerFunctionsBase):
    # toy example to highlight functionality of ServerFunctions.
    def __init__(self) -> None:
        # You can keep a state between different functions to have them work together.
        self.round = 0
        self.lr = 0.1

    # Skip any function to use the default FEDn implementation for the function.

    # Called first in the beggining of a round to select clients.
    def client_selection(self, client_ids: List[str]) -> List:
        # Pick 10 random clients
        client_ids = random.sample(client_ids, min(len(client_ids), 10))  # noqa: F405
        return client_ids

    # Called secondly before sending the global model.
    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        # Decrease learning rate every 10 rounds
        if self.round % 10 == 0:
            self.lr = self.lr * 0.1
        # see client/train.py for how to load the client settings.
        self.round += 1
        return {"learning_rate": self.lr}

    # Called third to aggregate the client updates.
    def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        # Weighted fedavg implementation.
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        total_weight = 0
        for client_id, (client_parameters, metadata) in client_updates.items():
            num_examples = metadata.get("num_examples", 1)
            total_weight += num_examples
            for i in range(len(weighted_sum)):
                weighted_sum[i] += client_parameters[i] * num_examples

        print("Models aggregated")
        averaged_updates = [weighted / total_weight for weighted in weighted_sum]
        return averaged_updates
