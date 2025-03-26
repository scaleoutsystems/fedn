import math
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from fedn.common.exceptions import InvalidParameterError
from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase
from fedn.utils.helpers.helperbase import HelperBase
from fedn.utils.parameters import Parameters


class Aggregator(AggregatorBase):
    """Federated Optimization (FedOpt) aggregator.

    Implmentation following: https://arxiv.org/pdf/2003.00295.pdf

    This aggregator computes pseudo gradients by subtracting the model
    update from the global model weights from the previous round.
    A server-side scheme is then applied, currenty supported schemes
    are "adam", "yogi", "adagrad".

    Limitations:
        - Only supports one combiner.
        - Momentum is reser for each new invokation of a training session.

    :param control: A handle to the :class: `fedn.network.combiner.updatehandler.UpdateHandler`
    :type control: class: `fedn.network.combiner.updatehandler.UpdateHandler`

    """

    def __init__(self, update_handler):
        super().__init__(update_handler)

        self.name = "fedopt"
        # To store momentum
        self.v = None
        self.m = None

    def combine_models(
        self, helper: Optional[HelperBase] = None, delete_models: bool = True, parameters: Optional[Parameters] = None
    ) -> Tuple[Optional[Any], Dict[str, float]]:
        """Compute pseudo gradients using model updates in the queue.

        :param helper: ML framework-specific helper, defaults to None.
        :param delete_models: Delete models from storage after aggregation, defaults to True.
        :param parameters: Aggregator hyperparameters, defaults to None.
        :return: The global model and metadata.
        """
        data = {"time_model_load": 0.0, "time_model_aggregation": 0.0}

        # Default hyperparameters
        default_parameters = {
            "serveropt": "adam",
            "learning_rate": 1e-3,
            "beta1": 0.9,
            "beta2": 0.99,
            "tau": 1e-4,
        }

        # Validate and merge parameters
        try:
            parameters = self._validate_and_merge_parameters(parameters, default_parameters)
        except InvalidParameterError as e:
            logger.error(f"Aggregator {self.name} received invalid parameters: {e}")
            return None, data

        logger.info(f"Aggregator {self.name} starting model aggregation.")

        # Aggregation initialization
        pseudo_gradient, model_old = None, None
        nr_aggregated_models, total_examples = 0, 0

        while not self.update_handler.model_updates.empty():
            try:
                logger.info(f"Aggregator {self.name}: Fetching next model update.")
                model_update = self.update_handler.next_model_update()

                tic = time.time()
                model_next, metadata = self.update_handler.load_model_update(model_update, helper)
                data["time_model_load"] += time.time() - tic

                logger.info(f"Processing model update {model_update.model_update_id}")

                # Increment total examples
                total_examples += metadata["num_examples"]

                tic = time.time()
                if nr_aggregated_models == 0:
                    model_old = self.update_handler.load_model(helper, model_update.model_id)
                    pseudo_gradient = helper.subtract(model_next, model_old)
                else:
                    pseudo_gradient_next = helper.subtract(model_next, model_old)
                    pseudo_gradient = helper.increment_average(pseudo_gradient, pseudo_gradient_next, metadata["num_examples"], total_examples)

                data["time_model_aggregation"] += time.time() - tic

                nr_aggregated_models += 1

                if delete_models:
                    self.update_handler.delete_model(model_update)
                    logger.info(f"Deleted model update {model_update.model_update_id} from storage.")
            except Exception as e:
                logger.error(f"Error processing model update: {e}. Skipping this update.")
                logger.error(traceback.format_exc())
                continue

        data["nr_aggregated_models"] = nr_aggregated_models

        if pseudo_gradient:
            try:
                model = self._apply_server_optimizer(helper, pseudo_gradient, model_old, parameters)
            except Exception as e:
                logger.error(f"Error during model aggregation: {e}")
                logger.error(traceback.format_exc())
                return None, data
        else:
            return None, data

        logger.info(f"Aggregator {self.name} completed. Aggregated {nr_aggregated_models} models.")
        return model, data

    def _validate_and_merge_parameters(self, parameters: Optional[Parameters], default_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and merge default parameters."""
        parameter_schema = {
            "serveropt": str,
            "learning_rate": float,
            "beta1": float,
            "beta2": float,
            "tau": float,
        }
        if parameters:
            parameters.validate(parameter_schema)
        else:
            logger.info(f"Aggregator {self.name} using default parameters.")
            parameters = {}
        return {**default_parameters, **parameters}

    def _apply_server_optimizer(self, helper: HelperBase, pseudo_gradient: Any, model_old: Any, parameters: Dict[str, Any]) -> Any:  # noqa: ANN401
        """Apply the selected server optimizer to compute the new model."""
        optimizer_map = {
            "adam": self.serveropt_adam,
            "yogi": self.serveropt_yogi,
            "adagrad": self.serveropt_adagrad,
        }
        optimizer = optimizer_map.get(parameters["serveropt"])
        if not optimizer:
            raise ValueError(f"Unsupported server optimizer: {parameters['serveropt']}")
        return optimizer(helper, pseudo_gradient, model_old, parameters)

    def serveropt_adam(self, helper, pseudo_gradient, model_old, parameters):
        """Server side optimization, FedAdam.

        :param helper: instance of helper class.
        :type helper: Helper
        :param pseudo_gradient: The pseudo gradient.
        :type pseudo_gradient: As defined by helper.
        :param model_old: The current global model.
        :type model_old: As defined in helper.
        :param parameters: Hyperparamters for the aggregator.
        :type parameters: dict
        :return: new model weights.
        :rtype: as defined by helper.
        """
        beta1 = parameters["beta1"]
        beta2 = parameters["beta2"]
        learning_rate = parameters["learning_rate"]
        tau = parameters["tau"]

        if not self.v:
            self.v = helper.ones(pseudo_gradient, math.pow(tau, 2))

        if not self.m:
            self.m = helper.multiply(pseudo_gradient, [(1.0 - beta1)] * len(pseudo_gradient))
        else:
            self.m = helper.add(self.m, pseudo_gradient, beta1, (1.0 - beta1))

        p = helper.power(pseudo_gradient, 2)
        self.v = helper.add(self.v, p, beta2, (1.0 - beta2))

        sv = helper.add(helper.sqrt(self.v), helper.ones(self.v, tau))
        t = helper.divide(self.m, sv)
        model = helper.add(model_old, t, 1.0, learning_rate)

        return model

    def serveropt_yogi(self, helper, pseudo_gradient, model_old, parameters):
        """Server side optimization, FedYogi.

        :param helper: instance of helper class.
        :type helper: Helper
        :param pseudo_gradient: The pseudo gradient.
        :type pseudo_gradient: As defined by helper.
        :param model_old: The current global model.
        :type model_old: As defined in helper.
        :param parameters: Hyperparamters for the aggregator.
        :type parameters: dict
        :return: new model weights.
        :rtype: as defined by helper.
        """
        beta1 = parameters["beta1"]
        beta2 = parameters["beta2"]
        learning_rate = parameters["learning_rate"]
        tau = parameters["tau"]

        if not self.v:
            self.v = helper.ones(pseudo_gradient, math.pow(tau, 2))

        if not self.m:
            self.m = helper.multiply(pseudo_gradient, [(1.0 - beta1)] * len(pseudo_gradient))
        else:
            self.m = helper.add(self.m, pseudo_gradient, beta1, (1.0 - beta1))

        p = helper.power(pseudo_gradient, 2)
        s = helper.sign(helper.add(self.v, p, 1.0, -1.0))
        s = helper.multiply(s, p)
        self.v = helper.add(self.v, s, 1.0, -(1.0 - beta2))

        sv = helper.add(helper.sqrt(self.v), helper.ones(self.v, tau))
        t = helper.divide(self.m, sv)
        model = helper.add(model_old, t, 1.0, learning_rate)

        return model

    def serveropt_adagrad(self, helper, pseudo_gradient, model_old, parameters):
        """Server side optimization, FedAdam.

        :param helper: instance of helper class.
        :type helper: Helper
        :param pseudo_gradient: The pseudo gradient.
        :type pseudo_gradient: As defined by helper.
        :param model_old: The current global model.
        :type model_old: As defined in helper.
        :param parameters: Hyperparamters for the aggregator.
        :type parameters: dict
        :return: new model weights.
        :rtype: as defined by helper.
        """
        beta1 = parameters["beta1"]
        learning_rate = parameters["learning_rate"]
        tau = parameters["tau"]

        if not self.v:
            self.v = helper.ones(pseudo_gradient, math.pow(tau, 2))

        if not self.m:
            self.m = helper.multiply(pseudo_gradient, [(1.0 - beta1)] * len(pseudo_gradient))
        else:
            self.m = helper.add(self.m, pseudo_gradient, beta1, (1.0 - beta1))

        p = helper.power(pseudo_gradient, 2)
        self.v = helper.add(self.v, p, 1.0, 1.0)

        sv = helper.add(helper.sqrt(self.v), helper.ones(self.v, tau))
        t = helper.divide(self.m, sv)
        model = helper.add(model_old, t, 1.0, learning_rate)

        return model
