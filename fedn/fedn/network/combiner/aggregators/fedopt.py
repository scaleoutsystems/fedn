import ast
import math

from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.aggregatorbase import AggregatorBase


class Aggregator(AggregatorBase):
    """ Federated Optimization (FedOpt) aggregator.

    Implmentation following: https://arxiv.org/pdf/2003.00295.pdf

    Aggregate pseudo gradients computed by subtracting the model
    update from the global model weights from the previous round.

    :param id: A reference to id of :class: `fedn.network.combiner.Combiner`
    :type id: str
    :param storage: Model repository for :class: `fedn.network.combiner.Combiner`
    :type storage: class: `fedn.common.storage.s3.s3repo.S3ModelRepository`
    :param server: A handle to the Combiner class :class: `fedn.network.combiner.Combiner`
    :type server: class: `fedn.network.combiner.Combiner`
    :param modelservice: A handle to the model service :class: `fedn.network.combiner.modelservice.ModelService`
    :type modelservice: class: `fedn.network.combiner.modelservice.ModelService`
    :param control: A handle to the :class: `fedn.network.combiner.roundhandler.RoundHandler`
    :type control: class: `fedn.network.combiner.roundhandler.RoundHandler`

    """

    def __init__(self, storage, server, modelservice, round_handler):

        super().__init__(storage, server, modelservice, round_handler)

        self.name = "fedopt"
        self.v = None
        self.m = None

        # Server side default hyperparameters. Note that these may need fine tuning.
        self.default_params = {
            'serveropt': 'adam',
            'learning_rate': 1e-2,
            'beta1': 0.9,
            'beta2': 0.99,
            'tau': 1e-4,
        }

    def combine_models(self, helper=None, delete_models=True, params=None):
        """Compute pseudo gradients using model updates in the queue.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`, optional
        :param time_window: The time window for model aggregation, defaults to 180
        :type time_window: int, optional
        :param max_nr_models: The maximum number of updates aggregated, defaults to 100
        :type max_nr_models: int, optional
        :param delete_models: Delete models from storage after aggregation, defaults to True
        :type delete_models: bool, optional
        :param params: Additional key-word arguments.
        :type params: dict
        :return: The global model and metadata
        :rtype: tuple
        """

        print("PARAMS: {}".format(params), flush=True)
        params = ast.literal_eval(params)
        data = {}
        data['time_model_load'] = 0.0
        data['time_model_aggregation'] = 0.0

        # Override default hyperparameters:
        if params:
            for key, value in self.default_params.items():
                print(key, value, flush=True)
                if key not in params:
                    print(key, flush=True)
                    params[key] = value
        else:
            params = self.default_params

        print("PARAMS2: {}".format(params), flush=True)

        model = None
        nr_aggregated_models = 0
        total_examples = 0

        logger.info(
            "AGGREGATOR({}): Aggregating model updates... ".format(self.name))

        while not self.model_updates.empty():
            try:
                # Get next model from queue
                model_update = self.next_model_update()

                # Load model paratmeters and metadata
                model_next, metadata = self.load_model_update(model_update, helper)

                logger.info(
                    "AGGREGATOR({}): Processing model update {}, metadata: {}  ".format(self.name, model_update.model_update_id, metadata))
                logger.info("***** {}".format(model_update))

                # Increment total number of examples
                total_examples += metadata['num_examples']

                if nr_aggregated_models == 0:
                    model_old = self.round_handler.load_model_update(helper, model_update.model_id)
                    pseudo_gradient = helper.subtract(model_next, model_old)
                else:
                    pseudo_gradient_next = helper.subtract(model_next, model_old)
                    pseudo_gradient = helper.increment_average(
                        pseudo_gradient, pseudo_gradient_next, metadata['num_examples'], total_examples)

                logger.info("NORM PSEUDOGRADIENT: {}".format(helper.norm(pseudo_gradient)))

                nr_aggregated_models += 1
                # Delete model from storage
                if delete_models:
                    self.modelservice.temp_model_storage.delete(model_update.model_update_id)
                    logger.info(
                        "AGGREGATOR({}): Deleted model update {} from storage.".format(self.name, model_update.model_update_id))
                self.model_updates.task_done()
            except Exception as e:
                logger.error(
                    "AGGREGATOR({}): Error encoutered while processing model update {}, skipping this update.".format(self.name, e))
                self.model_updates.task_done()

        if params['serveropt'] == 'adam':
            model = self.serveropt_adam(helper, pseudo_gradient, model_old)
        elif params['serveropt'] == 'yogi':
            model = self.serveropt_yogi(helper, pseudo_gradient, model_old)
        elif params['serveropt'] == 'adagrad':
            model = self.serveropt_adagrad(helper, pseudo_gradient, model_old)
        else:
            raise

        data['nr_aggregated_models'] = nr_aggregated_models

        logger.info("AGGREGATOR({}): Aggregation completed, aggregated {} models.".format(self.name, nr_aggregated_models))
        return model, data

    def serveropt_adam(self, helper, pseudo_gradient, model_old, params):
        """ Server side optimization, FedAdam.

        :param helper: instance of helper class.
        :type helper: Helper
        :param pseudo_gradient: The pseudo gradient.
        :type pseudo_gradient: As defined by helper.
        :return: new model weights.
        :rtype: as defined by helper.
        """
        beta1 = params['beta1']
        beta2 = params['beta2']
        learning_rate = params['learning_rate']
        tau = params['tau']

        if not self.v:
            self.v = helper.ones(pseudo_gradient, math.pow(tau, 2))

        if not self.m:
            self.m = helper.multiply(pseudo_gradient, [(1.0-beta1)]*len(pseudo_gradient))
        else:
            self.m = helper.add(self.m, pseudo_gradient, beta1, (1.0-beta1))

        p = helper.power(pseudo_gradient, 2)
        self.v = helper.add(self.v, p, beta2, (1.0-beta2))

        sv = helper.add(helper.sqrt(self.v), helper.ones(self.v, tau))
        t = helper.divide(self.m, sv)
        model = helper.add(model_old, t, 1.0, learning_rate)

        return model

    def serveropt_yogi(self, helper, pseudo_gradient, model_old, params):
        """ Server side optimization, FedAdam.

        :param helper: instance of helper class.
        :type helper: Helper
        :param pseudo_gradient: The pseudo gradient.
        :type pseudo_gradient: As defined by helper.
        :return: new model weights.
        :rtype: as defined by helper.
        """

        beta1 = params['beta1']
        beta2 = params['beta2']
        learning_rate = params['learning_rate']
        tau = params['tau']

        if not self.v:
            self.v = helper.ones(pseudo_gradient, math.pow(tau, 2))

        if not self.m:
            self.m = helper.multiply(pseudo_gradient, [(1.0-beta1)]*len(pseudo_gradient))
        else:
            self.m = helper.add(self.m, pseudo_gradient, beta1, (1.0-beta1))

        p = helper.power(pseudo_gradient, 2)
        s = helper.sign(helper.add(self.v, p, 1.0, -1.0))
        s = helper.multiply(s, p)
        self.v = helper.add(self.v, s, 1.0, -(1.0-beta2))

        sv = helper.add(helper.sqrt(self.v), helper.ones(self.v, tau))
        t = helper.divide(self.m, sv)
        model = helper.add(model_old, t, 1.0, learning_rate)

        return model

    def serveropt_adagrad(self, helper, pseudo_gradient, model_old, params):
        """ Server side optimization, FedAdam.

        :param helper: instance of helper class.
        :type helper: Helper
        :param pseudo_gradient: The pseudo gradient.
        :type pseudo_gradient: As defined by helper.
        :return: new model weights.
        :rtype: as defined by helper.
        """

        beta1 = params['beta1']
        learning_rate = params['learning_rate']
        tau = params['tau']

        if not self.v:
            self.v = helper.ones(pseudo_gradient, math.pow(tau, 2))

        if not self.m:
            self.m = helper.multiply(pseudo_gradient, [(1.0-beta1)]*len(pseudo_gradient))
        else:
            self.m = helper.add(self.m, pseudo_gradient, beta1, (1.0-beta1))

        p = helper.power(pseudo_gradient, 2)
        self.v = helper.add(self.v, p, 1.0, 1.0)

        sv = helper.add(helper.sqrt(self.v), helper.ones(self.v, tau))
        t = helper.divide(self.m, sv)
        model = helper.add(model_old, t, 1.0, learning_rate)

        return model
