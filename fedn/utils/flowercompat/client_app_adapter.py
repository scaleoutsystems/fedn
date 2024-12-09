from typing import Tuple

from flwr.client import ClientApp
from flwr.common import (
    Context,
    EvaluateIns,
    FitIns,
    GetParametersIns,
    Message,
    MessageType,
    MessageTypeLegacy,
    Metadata,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.recordset_compat import (
    evaluateins_to_recordset,
    fitins_to_recordset,
    getparametersins_to_recordset,
    recordset_to_evaluateres,
    recordset_to_fitres,
    recordset_to_getparametersres,
)


class FlwrClientAppAdapter:
    """Flwr ClientApp wrapper."""

    def __init__(self, app: ClientApp) -> None:
        self.app = app

    def init_parameters(self, partition_id: int, config: dict = {}):
        # Construct a get_parameters message for the ClientApp
        message, context = self._construct_message(MessageTypeLegacy.GET_PARAMETERS, [], partition_id, config)
        # Call client app with train message
        client_return_message = self.app(message, context)
        # return NDArrays of clients parameters
        parameters = self._parse_get_parameters_message(client_return_message)
        if len(parameters) == 0:
            raise ValueError(
                "The 'parameters' list is empty. Ensure your flower \
                             client has implemented a get_parameters() function."
            )
        return parameters

    def train(self, parameters: NDArrays, partition_id: int, config: dict = {}):
        # Construct a train message for the ClientApp with given parameters
        message, context = self._construct_message(MessageType.TRAIN, parameters, partition_id, config)
        # Call client app with train message
        client_return_message = self.app(message, context)
        # Parse return message
        params, num_examples = self._parse_train_message(client_return_message)
        return params, num_examples

    def evaluate(self, parameters: NDArrays, partition_id: int, config: dict = {}):
        # Construct an evaluate message for the ClientApp with given parameters
        message, context = self._construct_message(MessageType.EVALUATE, parameters, partition_id, config)
        # Call client app with evaluate message
        client_return_message = self.app(message, context)
        # Parse return message
        loss, accuracy = self._parse_evaluate_message(client_return_message)
        return loss, accuracy

    def _parse_get_parameters_message(self, message: Message) -> NDArrays:
        get_parameters_res = recordset_to_getparametersres(message.content, keep_input=False)
        return parameters_to_ndarrays(get_parameters_res.parameters)

    def _parse_train_message(self, message: Message) -> Tuple[NDArrays, int]:
        fitres = recordset_to_fitres(message.content, keep_input=False)
        params, num_examples = (
            parameters_to_ndarrays(fitres.parameters),
            fitres.num_examples,
        )
        return params, num_examples

    def _parse_evaluate_message(self, message: Message) -> Tuple[float, float]:
        evaluateres = recordset_to_evaluateres(message.content)
        return evaluateres.loss, evaluateres.metrics.get("accuracy", -1)

    def _construct_message(
        self,
        message_type: MessageType,
        parameters: NDArrays,
        partition_id: int,
        config: dict,
    ) -> Tuple[Message, Context]:
        parameters = ndarrays_to_parameters(parameters)
        if message_type == MessageType.TRAIN:
            fit_ins: FitIns = FitIns(parameters=parameters, config=config)
            recordset = fitins_to_recordset(fitins=fit_ins, keep_input=False)
        if message_type == MessageType.EVALUATE:
            ev_ins: EvaluateIns = EvaluateIns(parameters=parameters, config=config)
            recordset = evaluateins_to_recordset(evaluateins=ev_ins, keep_input=False)
        if message_type == MessageTypeLegacy.GET_PARAMETERS:
            get_parameters_ins: GetParametersIns = GetParametersIns(config=config)
            recordset = getparametersins_to_recordset(getparameters_ins=get_parameters_ins)

        metadata = Metadata(
            run_id=0,
            message_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message="",
            group_id="",
            ttl=0.0,
            message_type=message_type,
            partition_id=partition_id,
        )
        context = Context(recordset)
        message = Message(metadata=metadata, content=recordset)
        return message, context
