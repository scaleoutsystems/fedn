import numpy as np

from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.functionproviderbase import FunctionProviderBase


class FunctionProvider(FunctionProviderBase):
    """A FunctionProvider class responsible for aggregating client model parameters and performing
    hyperparameter tuning by adjusting the learning rate every 20th round. The class logs the current state of
    the model, learning rate, and round to facilitate monitoring and evaluation.
    """

    def __init__(self) -> None:
        self.current_round = -1
        self.initial_parameters = None
        self.learning_rates = [0.001, 0.01, 0.0001, 0.1, 0.00001]
        self.current_lr_index = -1
        self.current_lr = 0  # start with 0 learning rate the first round to get initial parameters
        self.current_parameters = None

        # Tracking metrics
        self.highest_accuracy = 0
        self.highest_accuracy_round = -1
        self.highest_accuracy_lr = 0
        self.mean_loss_per_lr = []
        self.mean_acc_per_lr = []
        self.highest_mean_acc = 0
        self.highest_mean_acc_round = -1
        self.highest_mean_acc_lr = None

    def aggregate(self, results: list[tuple[list[np.ndarray], dict]]) -> list[np.ndarray]:
        """Aggregate model parameters using weighted average based on the number of examples each client has.

        Args:
        ----
            results (list of tuples): Each tuple contains:
                - A list of numpy.ndarrays representing model parameters from a client.
                - A dictionary containing client metadata, which must include a key "num_examples" indicating
                  the number of examples used by the client.

        Returns:
        -------
            list of numpy.ndarrays: Aggregated model parameters as a list of numpy.ndarrays.

        """
        total_loss = 0
        total_acc = 0
        num_clients = len(results)
        if self.current_round == -1:
            self.initial_parameters = results[0][0]
            averaged_parameters = self.initial_parameters  # first round no updates were made.
        elif self.current_round % 20 == 0:
            if self.mean_loss_per_lr:
                logger.info(f"Completed Learning Rate: {self.current_lr}")
                logger.info(f"Mean Loss: {np.mean(self.mean_loss_per_lr)}, Highest Accuracy: {np.max(self.mean_acc_per_lr)}")
                logger.info(
                    f"""Highest mean accuracy across rounds: {self.highest_mean_acc}
                    at round {self.highest_mean_acc_round} with lr {self.highest_mean_acc_lr}"""
                )

            # Reset tracking for the new learning rate
            self.mean_loss_per_lr = []
            self.mean_acc_per_lr = []

            averaged_parameters = self.initial_parameters
            self.current_lr_index += 1
            self.current_lr = self.learning_rates[self.current_lr_index]
        else:
            # Aggregate using fedavg
            summed_parameters = [np.zeros_like(param) for param in results[0][0]]
            total_weight = 0
            for client_params, client_metadata in results:
                weight = client_metadata.get("num_examples", 1)
                total_weight += weight
                for i, param in enumerate(client_params):
                    summed_parameters[i] += param * weight

                total_loss += client_metadata.get("test_loss", 0)
                total_acc += client_metadata.get("test_acc", 0)

            averaged_parameters = [param / total_weight for param in summed_parameters]

        # Calculate average loss and accuracy by number of clients
        avg_loss = total_loss / num_clients if num_clients > 0 else 0
        avg_acc = total_acc / num_clients if num_clients > 0 else 0

        # Update the tracking for the current learning rate
        self.mean_loss_per_lr.append(avg_loss)
        self.mean_acc_per_lr.append(avg_acc)

        # Check if we have a new highest accuracy
        if avg_acc > self.highest_accuracy:
            self.highest_accuracy = avg_acc
            self.highest_accuracy_round = self.current_round
            self.highest_accuracy_lr = self.current_lr

        # Check if we have a new highest mean accuracy across rounds
        if avg_acc > self.highest_mean_acc:
            self.highest_mean_acc = avg_acc
            self.highest_mean_acc_round = self.current_round
            self.highest_mean_acc_lr = self.current_lr

        # Print the metrics
        logger.info(f"Round {self.current_round} - Learning Rate: {self.current_lr}")
        logger.info(f"Average Test Loss: {avg_loss}, Average Test Accuracy: {avg_acc}")
        logger.info(f"Highest Accuracy Achieved: {self.highest_accuracy} at round {self.highest_accuracy_round} with lr {self.highest_accuracy_lr}")

        self.current_round += 1
        return averaged_parameters

    def get_model_metadata(self):
        return {"learning_rate": self.current_lr, "parameter_tuning": True}
