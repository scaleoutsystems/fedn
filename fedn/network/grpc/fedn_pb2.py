# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: fedn.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'fedn.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nfedn.proto\x12\x04\x66\x65\x64n\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1egoogle/protobuf/wrappers.proto\":\n\x08Response\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08response\x18\x02 \x01(\t\"\xf1\x01\n\x06Status\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x0e\n\x06status\x18\x02 \x01(\t\x12!\n\tlog_level\x18\x03 \x01(\x0e\x32\x0e.fedn.LogLevel\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x1e\n\x04type\x18\x07 \x01(\x0e\x32\x10.fedn.StatusType\x12\r\n\x05\x65xtra\x18\x08 \x01(\t\x12\x12\n\nsession_id\x18\t \x01(\t\"\xd8\x01\n\x0bTaskRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12\x11\n\ttimestamp\x18\x06 \x01(\t\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x12\n\nsession_id\x18\x08 \x01(\t\x12\x1e\n\x04type\x18\t \x01(\x0e\x32\x10.fedn.StatusType\"\xbf\x01\n\x0bModelUpdate\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x17\n\x0fmodel_update_id\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12\x11\n\ttimestamp\x18\x06 \x01(\t\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x08 \x01(\t\"\xd8\x01\n\x0fModelValidation\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x12\n\nsession_id\x18\x08 \x01(\t\"\xdb\x01\n\x0fModelPrediction\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x15\n\rprediction_id\x18\x08 \x01(\t\"\xd0\x01\n\x12\x42\x61\x63kwardCompletion\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x13\n\x0bgradient_id\x18\x03 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x04 \x01(\t\x12\x12\n\nsession_id\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04meta\x18\x07 \x01(\t\"\xe1\x01\n\x0bModelMetric\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12!\n\x07metrics\x18\x02 \x03(\x0b\x32\x10.fedn.MetricElem\x12-\n\ttimestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12*\n\x04step\x18\x04 \x01(\x0b\x32\x1c.google.protobuf.UInt32Value\x12\x10\n\x08model_id\x18\x05 \x01(\t\x12\x10\n\x08round_id\x18\x06 \x01(\t\x12\x12\n\nsession_id\x18\x07 \x01(\t\"(\n\nMetricElem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02\"\x88\x01\n\x10\x41ttributeMessage\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\'\n\nattributes\x18\x02 \x03(\x0b\x32\x13.fedn.AttributeElem\x12-\n\ttimestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"+\n\rAttributeElem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"\x89\x01\n\x10TelemetryMessage\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12(\n\x0btelemetries\x18\x02 \x03(\x0b\x32\x13.fedn.TelemetryElem\x12-\n\ttimestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"+\n\rTelemetryElem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02\"\x89\x01\n\x0cModelRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\n\n\x02id\x18\x04 \x01(\t\x12!\n\x06status\x18\x05 \x01(\x0e\x32\x11.fedn.ModelStatus\"]\n\rModelResponse\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\n\n\x02id\x18\x02 \x01(\t\x12!\n\x06status\x18\x03 \x01(\x0e\x32\x11.fedn.ModelStatus\x12\x0f\n\x07message\x18\x04 \x01(\t\"U\n\x15GetGlobalModelRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\"h\n\x16GetGlobalModelResponse\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\"^\n\tHeartbeat\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1a\n\x12memory_utilisation\x18\x02 \x01(\x02\x12\x17\n\x0f\x63pu_utilisation\x18\x03 \x01(\x02\"W\n\x16\x43lientAvailableMessage\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t\x12\x11\n\ttimestamp\x18\x03 \x01(\t\"P\n\x12ListClientsRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1c\n\x07\x63hannel\x18\x02 \x01(\x0e\x32\x0b.fedn.Queue\"*\n\nClientList\x12\x1c\n\x06\x63lient\x18\x01 \x03(\x0b\x32\x0c.fedn.Client\"C\n\x06\x43lient\x12\x18\n\x04role\x18\x01 \x01(\x0e\x32\n.fedn.Role\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x11\n\tclient_id\x18\x03 \x01(\t\"m\n\x0fReassignRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x0e\n\x06server\x18\x03 \x01(\t\x12\x0c\n\x04port\x18\x04 \x01(\r\"c\n\x10ReconnectRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x11\n\treconnect\x18\x03 \x01(\r\"\'\n\tParameter\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"T\n\x0e\x43ontrolRequest\x12\x1e\n\x07\x63ommand\x18\x01 \x01(\x0e\x32\r.fedn.Command\x12\"\n\tparameter\x18\x02 \x03(\x0b\x32\x0f.fedn.Parameter\"F\n\x0f\x43ontrolResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\"\n\tparameter\x18\x02 \x03(\x0b\x32\x0f.fedn.Parameter\"\x13\n\x11\x43onnectionRequest\"<\n\x12\x43onnectionResponse\x12&\n\x06status\x18\x01 \x01(\x0e\x32\x16.fedn.ConnectionStatus\"1\n\x18ProvidedFunctionsRequest\x12\x15\n\rfunction_code\x18\x01 \x01(\t\"\xac\x01\n\x19ProvidedFunctionsResponse\x12T\n\x13\x61vailable_functions\x18\x01 \x03(\x0b\x32\x37.fedn.ProvidedFunctionsResponse.AvailableFunctionsEntry\x1a\x39\n\x17\x41vailableFunctionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\"#\n\x13\x43lientConfigRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"/\n\x14\x43lientConfigResponse\x12\x17\n\x0f\x63lient_settings\x18\x01 \x01(\t\",\n\x16\x43lientSelectionRequest\x12\x12\n\nclient_ids\x18\x01 \x01(\t\"-\n\x17\x43lientSelectionResponse\x12\x12\n\nclient_ids\x18\x01 \x01(\t\"8\n\x11\x43lientMetaRequest\x12\x10\n\x08metadata\x18\x01 \x01(\t\x12\x11\n\tclient_id\x18\x02 \x01(\t\"$\n\x12\x43lientMetaResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"-\n\x11StoreModelRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\n\n\x02id\x18\x02 \x01(\t\"$\n\x12StoreModelResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"\'\n\x12\x41ggregationRequest\x12\x11\n\taggregate\x18\x01 \x01(\t\"#\n\x13\x41ggregationResponse\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c*\xde\x01\n\nStatusType\x12\x07\n\x03LOG\x10\x00\x12\x18\n\x14MODEL_UPDATE_REQUEST\x10\x01\x12\x10\n\x0cMODEL_UPDATE\x10\x02\x12\x1c\n\x18MODEL_VALIDATION_REQUEST\x10\x03\x12\x14\n\x10MODEL_VALIDATION\x10\x04\x12\x14\n\x10MODEL_PREDICTION\x10\x05\x12\x0b\n\x07NETWORK\x10\x06\x12\x13\n\x0f\x46ORWARD_REQUEST\x10\x07\x12\x0b\n\x07\x46ORWARD\x10\x08\x12\x14\n\x10\x42\x41\x43KWARD_REQUEST\x10\t\x12\x0c\n\x08\x42\x41\x43KWARD\x10\n*L\n\x08LogLevel\x12\x08\n\x04NONE\x10\x00\x12\x08\n\x04INFO\x10\x01\x12\t\n\x05\x44\x45\x42UG\x10\x02\x12\x0b\n\x07WARNING\x10\x03\x12\t\n\x05\x45RROR\x10\x04\x12\t\n\x05\x41UDIT\x10\x05*$\n\x05Queue\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x0e\n\nTASK_QUEUE\x10\x01*S\n\x0bModelStatus\x12\x06\n\x02OK\x10\x00\x12\x0f\n\x0bIN_PROGRESS\x10\x01\x12\x12\n\x0eIN_PROGRESS_OK\x10\x02\x12\n\n\x06\x46\x41ILED\x10\x03\x12\x0b\n\x07UNKNOWN\x10\x04*8\n\x04Role\x12\t\n\x05OTHER\x10\x00\x12\n\n\x06\x43LIENT\x10\x01\x12\x0c\n\x08\x43OMBINER\x10\x02\x12\x0b\n\x07REDUCER\x10\x03*J\n\x07\x43ommand\x12\x08\n\x04IDLE\x10\x00\x12\t\n\x05START\x10\x01\x12\t\n\x05PAUSE\x10\x02\x12\x08\n\x04STOP\x10\x03\x12\t\n\x05RESET\x10\x04\x12\n\n\x06REPORT\x10\x05*I\n\x10\x43onnectionStatus\x12\x11\n\rNOT_ACCEPTING\x10\x00\x12\r\n\tACCEPTING\x10\x01\x12\x13\n\x0fTRY_AGAIN_LATER\x10\x02\x32z\n\x0cModelService\x12\x33\n\x06Upload\x12\x12.fedn.ModelRequest\x1a\x13.fedn.ModelResponse(\x01\x12\x35\n\x08\x44ownload\x12\x12.fedn.ModelRequest\x1a\x13.fedn.ModelResponse0\x01\x32\xbb\x02\n\x07\x43ontrol\x12\x34\n\x05Start\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12\x33\n\x04Stop\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12\x44\n\x15\x46lushAggregationQueue\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12<\n\rSetAggregator\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12\x41\n\x12SetServerFunctions\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse2V\n\x07Reducer\x12K\n\x0eGetGlobalModel\x12\x1b.fedn.GetGlobalModelRequest\x1a\x1c.fedn.GetGlobalModelResponse2\xab\x03\n\tConnector\x12\x44\n\x14\x41llianceStatusStream\x12\x1c.fedn.ClientAvailableMessage\x1a\x0c.fedn.Status0\x01\x12*\n\nSendStatus\x12\x0c.fedn.Status\x1a\x0e.fedn.Response\x12?\n\x11ListActiveClients\x12\x18.fedn.ListClientsRequest\x1a\x10.fedn.ClientList\x12\x45\n\x10\x41\x63\x63\x65ptingClients\x12\x17.fedn.ConnectionRequest\x1a\x18.fedn.ConnectionResponse\x12\x30\n\rSendHeartbeat\x12\x0f.fedn.Heartbeat\x1a\x0e.fedn.Response\x12\x37\n\x0eReassignClient\x12\x15.fedn.ReassignRequest\x1a\x0e.fedn.Response\x12\x39\n\x0fReconnectClient\x12\x16.fedn.ReconnectRequest\x1a\x0e.fedn.Response2\xf7\x03\n\x08\x43ombiner\x12?\n\nTaskStream\x12\x1c.fedn.ClientAvailableMessage\x1a\x11.fedn.TaskRequest0\x01\x12\x34\n\x0fSendModelUpdate\x12\x11.fedn.ModelUpdate\x1a\x0e.fedn.Response\x12<\n\x13SendModelValidation\x12\x15.fedn.ModelValidation\x1a\x0e.fedn.Response\x12<\n\x13SendModelPrediction\x12\x15.fedn.ModelPrediction\x1a\x0e.fedn.Response\x12\x42\n\x16SendBackwardCompletion\x12\x18.fedn.BackwardCompletion\x1a\x0e.fedn.Response\x12\x34\n\x0fSendModelMetric\x12\x11.fedn.ModelMetric\x1a\x0e.fedn.Response\x12>\n\x14SendAttributeMessage\x12\x16.fedn.AttributeMessage\x1a\x0e.fedn.Response\x12>\n\x14SendTelemetryMessage\x12\x16.fedn.TelemetryMessage\x1a\x0e.fedn.Response2\xec\x03\n\x0f\x46unctionService\x12Z\n\x17HandleProvidedFunctions\x12\x1e.fedn.ProvidedFunctionsRequest\x1a\x1f.fedn.ProvidedFunctionsResponse\x12M\n\x12HandleClientConfig\x12\x19.fedn.ClientConfigRequest\x1a\x1a.fedn.ClientConfigResponse(\x01\x12T\n\x15HandleClientSelection\x12\x1c.fedn.ClientSelectionRequest\x1a\x1d.fedn.ClientSelectionResponse\x12\x43\n\x0eHandleMetadata\x12\x17.fedn.ClientMetaRequest\x1a\x18.fedn.ClientMetaResponse\x12G\n\x10HandleStoreModel\x12\x17.fedn.StoreModelRequest\x1a\x18.fedn.StoreModelResponse(\x01\x12J\n\x11HandleAggregation\x12\x18.fedn.AggregationRequest\x1a\x19.fedn.AggregationResponse0\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'fedn_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._loaded_options = None
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._serialized_options = b'8\001'
  _globals['_STATUSTYPE']._serialized_start=4060
  _globals['_STATUSTYPE']._serialized_end=4282
  _globals['_LOGLEVEL']._serialized_start=4284
  _globals['_LOGLEVEL']._serialized_end=4360
  _globals['_QUEUE']._serialized_start=4362
  _globals['_QUEUE']._serialized_end=4398
  _globals['_MODELSTATUS']._serialized_start=4400
  _globals['_MODELSTATUS']._serialized_end=4483
  _globals['_ROLE']._serialized_start=4485
  _globals['_ROLE']._serialized_end=4541
  _globals['_COMMAND']._serialized_start=4543
  _globals['_COMMAND']._serialized_end=4617
  _globals['_CONNECTIONSTATUS']._serialized_start=4619
  _globals['_CONNECTIONSTATUS']._serialized_end=4692
  _globals['_RESPONSE']._serialized_start=85
  _globals['_RESPONSE']._serialized_end=143
  _globals['_STATUS']._serialized_start=146
  _globals['_STATUS']._serialized_end=387
  _globals['_TASKREQUEST']._serialized_start=390
  _globals['_TASKREQUEST']._serialized_end=606
  _globals['_MODELUPDATE']._serialized_start=609
  _globals['_MODELUPDATE']._serialized_end=800
  _globals['_MODELVALIDATION']._serialized_start=803
  _globals['_MODELVALIDATION']._serialized_end=1019
  _globals['_MODELPREDICTION']._serialized_start=1022
  _globals['_MODELPREDICTION']._serialized_end=1241
  _globals['_BACKWARDCOMPLETION']._serialized_start=1244
  _globals['_BACKWARDCOMPLETION']._serialized_end=1452
  _globals['_MODELMETRIC']._serialized_start=1455
  _globals['_MODELMETRIC']._serialized_end=1680
  _globals['_METRICELEM']._serialized_start=1682
  _globals['_METRICELEM']._serialized_end=1722
  _globals['_ATTRIBUTEMESSAGE']._serialized_start=1725
  _globals['_ATTRIBUTEMESSAGE']._serialized_end=1861
  _globals['_ATTRIBUTEELEM']._serialized_start=1863
  _globals['_ATTRIBUTEELEM']._serialized_end=1906
  _globals['_TELEMETRYMESSAGE']._serialized_start=1909
  _globals['_TELEMETRYMESSAGE']._serialized_end=2046
  _globals['_TELEMETRYELEM']._serialized_start=2048
  _globals['_TELEMETRYELEM']._serialized_end=2091
  _globals['_MODELREQUEST']._serialized_start=2094
  _globals['_MODELREQUEST']._serialized_end=2231
  _globals['_MODELRESPONSE']._serialized_start=2233
  _globals['_MODELRESPONSE']._serialized_end=2326
  _globals['_GETGLOBALMODELREQUEST']._serialized_start=2328
  _globals['_GETGLOBALMODELREQUEST']._serialized_end=2413
  _globals['_GETGLOBALMODELRESPONSE']._serialized_start=2415
  _globals['_GETGLOBALMODELRESPONSE']._serialized_end=2519
  _globals['_HEARTBEAT']._serialized_start=2521
  _globals['_HEARTBEAT']._serialized_end=2615
  _globals['_CLIENTAVAILABLEMESSAGE']._serialized_start=2617
  _globals['_CLIENTAVAILABLEMESSAGE']._serialized_end=2704
  _globals['_LISTCLIENTSREQUEST']._serialized_start=2706
  _globals['_LISTCLIENTSREQUEST']._serialized_end=2786
  _globals['_CLIENTLIST']._serialized_start=2788
  _globals['_CLIENTLIST']._serialized_end=2830
  _globals['_CLIENT']._serialized_start=2832
  _globals['_CLIENT']._serialized_end=2899
  _globals['_REASSIGNREQUEST']._serialized_start=2901
  _globals['_REASSIGNREQUEST']._serialized_end=3010
  _globals['_RECONNECTREQUEST']._serialized_start=3012
  _globals['_RECONNECTREQUEST']._serialized_end=3111
  _globals['_PARAMETER']._serialized_start=3113
  _globals['_PARAMETER']._serialized_end=3152
  _globals['_CONTROLREQUEST']._serialized_start=3154
  _globals['_CONTROLREQUEST']._serialized_end=3238
  _globals['_CONTROLRESPONSE']._serialized_start=3240
  _globals['_CONTROLRESPONSE']._serialized_end=3310
  _globals['_CONNECTIONREQUEST']._serialized_start=3312
  _globals['_CONNECTIONREQUEST']._serialized_end=3331
  _globals['_CONNECTIONRESPONSE']._serialized_start=3333
  _globals['_CONNECTIONRESPONSE']._serialized_end=3393
  _globals['_PROVIDEDFUNCTIONSREQUEST']._serialized_start=3395
  _globals['_PROVIDEDFUNCTIONSREQUEST']._serialized_end=3444
  _globals['_PROVIDEDFUNCTIONSRESPONSE']._serialized_start=3447
  _globals['_PROVIDEDFUNCTIONSRESPONSE']._serialized_end=3619
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._serialized_start=3562
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._serialized_end=3619
  _globals['_CLIENTCONFIGREQUEST']._serialized_start=3621
  _globals['_CLIENTCONFIGREQUEST']._serialized_end=3656
  _globals['_CLIENTCONFIGRESPONSE']._serialized_start=3658
  _globals['_CLIENTCONFIGRESPONSE']._serialized_end=3705
  _globals['_CLIENTSELECTIONREQUEST']._serialized_start=3707
  _globals['_CLIENTSELECTIONREQUEST']._serialized_end=3751
  _globals['_CLIENTSELECTIONRESPONSE']._serialized_start=3753
  _globals['_CLIENTSELECTIONRESPONSE']._serialized_end=3798
  _globals['_CLIENTMETAREQUEST']._serialized_start=3800
  _globals['_CLIENTMETAREQUEST']._serialized_end=3856
  _globals['_CLIENTMETARESPONSE']._serialized_start=3858
  _globals['_CLIENTMETARESPONSE']._serialized_end=3894
  _globals['_STOREMODELREQUEST']._serialized_start=3896
  _globals['_STOREMODELREQUEST']._serialized_end=3941
  _globals['_STOREMODELRESPONSE']._serialized_start=3943
  _globals['_STOREMODELRESPONSE']._serialized_end=3979
  _globals['_AGGREGATIONREQUEST']._serialized_start=3981
  _globals['_AGGREGATIONREQUEST']._serialized_end=4020
  _globals['_AGGREGATIONRESPONSE']._serialized_start=4022
  _globals['_AGGREGATIONRESPONSE']._serialized_end=4057
  _globals['_MODELSERVICE']._serialized_start=4694
  _globals['_MODELSERVICE']._serialized_end=4816
  _globals['_CONTROL']._serialized_start=4819
  _globals['_CONTROL']._serialized_end=5134
  _globals['_REDUCER']._serialized_start=5136
  _globals['_REDUCER']._serialized_end=5222
  _globals['_CONNECTOR']._serialized_start=5225
  _globals['_CONNECTOR']._serialized_end=5652
  _globals['_COMBINER']._serialized_start=5655
  _globals['_COMBINER']._serialized_end=6158
  _globals['_FUNCTIONSERVICE']._serialized_start=6161
  _globals['_FUNCTIONSERVICE']._serialized_end=6653
# @@protoc_insertion_point(module_scope)
