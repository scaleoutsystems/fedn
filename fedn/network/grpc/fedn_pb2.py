# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: network/grpc/fedn.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'network/grpc/fedn.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17network/grpc/fedn.proto\x12\x04\x66\x65\x64n\x1a\x1fgoogle/protobuf/timestamp.proto\":\n\x08Response\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08response\x18\x02 \x01(\t\"\xbc\x02\n\x06Status\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x0e\n\x06status\x18\x02 \x01(\t\x12(\n\tlog_level\x18\x03 \x01(\x0e\x32\x15.fedn.Status.LogLevel\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x1e\n\x04type\x18\x07 \x01(\x0e\x32\x10.fedn.StatusType\x12\r\n\x05\x65xtra\x18\x08 \x01(\t\x12\x12\n\nsession_id\x18\t \x01(\t\"B\n\x08LogLevel\x12\x08\n\x04INFO\x10\x00\x12\t\n\x05\x44\x45\x42UG\x10\x01\x12\x0b\n\x07WARNING\x10\x02\x12\t\n\x05\x45RROR\x10\x03\x12\t\n\x05\x41UDIT\x10\x04\"\xd8\x01\n\x0bTaskRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12\x11\n\ttimestamp\x18\x06 \x01(\t\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x12\n\nsession_id\x18\x08 \x01(\t\x12\x1e\n\x04type\x18\t \x01(\x0e\x32\x10.fedn.StatusType\"\xbf\x01\n\x0bModelUpdate\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x17\n\x0fmodel_update_id\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12\x11\n\ttimestamp\x18\x06 \x01(\t\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x08 \x01(\t\"\xd8\x01\n\x0fModelValidation\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x12\n\nsession_id\x18\x08 \x01(\t\"\xdb\x01\n\x0fModelPrediction\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\t\x12\x16\n\x0e\x63orrelation_id\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04meta\x18\x07 \x01(\t\x12\x15\n\rprediction_id\x18\x08 \x01(\t\"\x89\x01\n\x0cModelRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\n\n\x02id\x18\x04 \x01(\t\x12!\n\x06status\x18\x05 \x01(\x0e\x32\x11.fedn.ModelStatus\"]\n\rModelResponse\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\n\n\x02id\x18\x02 \x01(\t\x12!\n\x06status\x18\x03 \x01(\x0e\x32\x11.fedn.ModelStatus\x12\x0f\n\x07message\x18\x04 \x01(\t\"U\n\x15GetGlobalModelRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\"h\n\x16GetGlobalModelResponse\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x10\n\x08model_id\x18\x03 \x01(\t\")\n\tHeartbeat\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\"W\n\x16\x43lientAvailableMessage\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t\x12\x11\n\ttimestamp\x18\x03 \x01(\t\"P\n\x12ListClientsRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1c\n\x07\x63hannel\x18\x02 \x01(\x0e\x32\x0b.fedn.Queue\"*\n\nClientList\x12\x1c\n\x06\x63lient\x18\x01 \x03(\x0b\x32\x0c.fedn.Client\"C\n\x06\x43lient\x12\x18\n\x04role\x18\x01 \x01(\x0e\x32\n.fedn.Role\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x11\n\tclient_id\x18\x03 \x01(\t\"m\n\x0fReassignRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x0e\n\x06server\x18\x03 \x01(\t\x12\x0c\n\x04port\x18\x04 \x01(\r\"c\n\x10ReconnectRequest\x12\x1c\n\x06sender\x18\x01 \x01(\x0b\x32\x0c.fedn.Client\x12\x1e\n\x08receiver\x18\x02 \x01(\x0b\x32\x0c.fedn.Client\x12\x11\n\treconnect\x18\x03 \x01(\r\"\'\n\tParameter\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"T\n\x0e\x43ontrolRequest\x12\x1e\n\x07\x63ommand\x18\x01 \x01(\x0e\x32\r.fedn.Command\x12\"\n\tparameter\x18\x02 \x03(\x0b\x32\x0f.fedn.Parameter\"F\n\x0f\x43ontrolResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\"\n\tparameter\x18\x02 \x03(\x0b\x32\x0f.fedn.Parameter\"\x13\n\x11\x43onnectionRequest\"<\n\x12\x43onnectionResponse\x12&\n\x06status\x18\x01 \x01(\x0e\x32\x16.fedn.ConnectionStatus\"1\n\x18ProvidedFunctionsRequest\x12\x15\n\rfunction_code\x18\x01 \x01(\t\"\xac\x01\n\x19ProvidedFunctionsResponse\x12T\n\x13\x61vailable_functions\x18\x01 \x03(\x0b\x32\x37.fedn.ProvidedFunctionsResponse.AvailableFunctionsEntry\x1a\x39\n\x17\x41vailableFunctionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\"#\n\x13\x43lientConfigRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"/\n\x14\x43lientConfigResponse\x12\x17\n\x0f\x63lient_settings\x18\x01 \x01(\t\",\n\x16\x43lientSelectionRequest\x12\x12\n\nclient_ids\x18\x01 \x01(\t\"-\n\x17\x43lientSelectionResponse\x12\x12\n\nclient_ids\x18\x01 \x01(\t\"8\n\x11\x43lientMetaRequest\x12\x10\n\x08metadata\x18\x01 \x01(\t\x12\x11\n\tclient_id\x18\x02 \x01(\t\"$\n\x12\x43lientMetaResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"-\n\x11StoreModelRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\n\n\x02id\x18\x02 \x01(\t\"$\n\x12StoreModelResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"\'\n\x12\x41ggregationRequest\x12\x11\n\taggregate\x18\x01 \x01(\t\"#\n\x13\x41ggregationResponse\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c*\x8b\x01\n\nStatusType\x12\x07\n\x03LOG\x10\x00\x12\x18\n\x14MODEL_UPDATE_REQUEST\x10\x01\x12\x10\n\x0cMODEL_UPDATE\x10\x02\x12\x1c\n\x18MODEL_VALIDATION_REQUEST\x10\x03\x12\x14\n\x10MODEL_VALIDATION\x10\x04\x12\x14\n\x10MODEL_PREDICTION\x10\x05*$\n\x05Queue\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x0e\n\nTASK_QUEUE\x10\x01*S\n\x0bModelStatus\x12\x06\n\x02OK\x10\x00\x12\x0f\n\x0bIN_PROGRESS\x10\x01\x12\x12\n\x0eIN_PROGRESS_OK\x10\x02\x12\n\n\x06\x46\x41ILED\x10\x03\x12\x0b\n\x07UNKNOWN\x10\x04*8\n\x04Role\x12\n\n\x06WORKER\x10\x00\x12\x0c\n\x08\x43OMBINER\x10\x01\x12\x0b\n\x07REDUCER\x10\x02\x12\t\n\x05OTHER\x10\x03*J\n\x07\x43ommand\x12\x08\n\x04IDLE\x10\x00\x12\t\n\x05START\x10\x01\x12\t\n\x05PAUSE\x10\x02\x12\x08\n\x04STOP\x10\x03\x12\t\n\x05RESET\x10\x04\x12\n\n\x06REPORT\x10\x05*I\n\x10\x43onnectionStatus\x12\x11\n\rNOT_ACCEPTING\x10\x00\x12\r\n\tACCEPTING\x10\x01\x12\x13\n\x0fTRY_AGAIN_LATER\x10\x02\x32z\n\x0cModelService\x12\x33\n\x06Upload\x12\x12.fedn.ModelRequest\x1a\x13.fedn.ModelResponse(\x01\x12\x35\n\x08\x44ownload\x12\x12.fedn.ModelRequest\x1a\x13.fedn.ModelResponse0\x01\x32\xbb\x02\n\x07\x43ontrol\x12\x34\n\x05Start\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12\x33\n\x04Stop\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12\x44\n\x15\x46lushAggregationQueue\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12<\n\rSetAggregator\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse\x12\x41\n\x12SetServerFunctions\x12\x14.fedn.ControlRequest\x1a\x15.fedn.ControlResponse2V\n\x07Reducer\x12K\n\x0eGetGlobalModel\x12\x1b.fedn.GetGlobalModelRequest\x1a\x1c.fedn.GetGlobalModelResponse2\xab\x03\n\tConnector\x12\x44\n\x14\x41llianceStatusStream\x12\x1c.fedn.ClientAvailableMessage\x1a\x0c.fedn.Status0\x01\x12*\n\nSendStatus\x12\x0c.fedn.Status\x1a\x0e.fedn.Response\x12?\n\x11ListActiveClients\x12\x18.fedn.ListClientsRequest\x1a\x10.fedn.ClientList\x12\x45\n\x10\x41\x63\x63\x65ptingClients\x12\x17.fedn.ConnectionRequest\x1a\x18.fedn.ConnectionResponse\x12\x30\n\rSendHeartbeat\x12\x0f.fedn.Heartbeat\x1a\x0e.fedn.Response\x12\x37\n\x0eReassignClient\x12\x15.fedn.ReassignRequest\x1a\x0e.fedn.Response\x12\x39\n\x0fReconnectClient\x12\x16.fedn.ReconnectRequest\x1a\x0e.fedn.Response2\xfd\x01\n\x08\x43ombiner\x12?\n\nTaskStream\x12\x1c.fedn.ClientAvailableMessage\x1a\x11.fedn.TaskRequest0\x01\x12\x34\n\x0fSendModelUpdate\x12\x11.fedn.ModelUpdate\x1a\x0e.fedn.Response\x12<\n\x13SendModelValidation\x12\x15.fedn.ModelValidation\x1a\x0e.fedn.Response\x12<\n\x13SendModelPrediction\x12\x15.fedn.ModelPrediction\x1a\x0e.fedn.Response2\xec\x03\n\x0f\x46unctionService\x12Z\n\x17HandleProvidedFunctions\x12\x1e.fedn.ProvidedFunctionsRequest\x1a\x1f.fedn.ProvidedFunctionsResponse\x12M\n\x12HandleClientConfig\x12\x19.fedn.ClientConfigRequest\x1a\x1a.fedn.ClientConfigResponse(\x01\x12T\n\x15HandleClientSelection\x12\x1c.fedn.ClientSelectionRequest\x1a\x1d.fedn.ClientSelectionResponse\x12\x43\n\x0eHandleMetadata\x12\x17.fedn.ClientMetaRequest\x1a\x18.fedn.ClientMetaResponse\x12G\n\x10HandleStoreModel\x12\x17.fedn.StoreModelRequest\x1a\x18.fedn.StoreModelResponse(\x01\x12J\n\x11HandleAggregation\x12\x18.fedn.AggregationRequest\x1a\x19.fedn.AggregationResponse0\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'network.grpc.fedn_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._loaded_options = None
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._serialized_options = b'8\001'
  _globals['_STATUSTYPE']._serialized_start=3213
  _globals['_STATUSTYPE']._serialized_end=3352
  _globals['_QUEUE']._serialized_start=3354
  _globals['_QUEUE']._serialized_end=3390
  _globals['_MODELSTATUS']._serialized_start=3392
  _globals['_MODELSTATUS']._serialized_end=3475
  _globals['_ROLE']._serialized_start=3477
  _globals['_ROLE']._serialized_end=3533
  _globals['_COMMAND']._serialized_start=3535
  _globals['_COMMAND']._serialized_end=3609
  _globals['_CONNECTIONSTATUS']._serialized_start=3611
  _globals['_CONNECTIONSTATUS']._serialized_end=3684
  _globals['_RESPONSE']._serialized_start=66
  _globals['_RESPONSE']._serialized_end=124
  _globals['_STATUS']._serialized_start=127
  _globals['_STATUS']._serialized_end=443
  _globals['_STATUS_LOGLEVEL']._serialized_start=377
  _globals['_STATUS_LOGLEVEL']._serialized_end=443
  _globals['_TASKREQUEST']._serialized_start=446
  _globals['_TASKREQUEST']._serialized_end=662
  _globals['_MODELUPDATE']._serialized_start=665
  _globals['_MODELUPDATE']._serialized_end=856
  _globals['_MODELVALIDATION']._serialized_start=859
  _globals['_MODELVALIDATION']._serialized_end=1075
  _globals['_MODELPREDICTION']._serialized_start=1078
  _globals['_MODELPREDICTION']._serialized_end=1297
  _globals['_MODELREQUEST']._serialized_start=1300
  _globals['_MODELREQUEST']._serialized_end=1437
  _globals['_MODELRESPONSE']._serialized_start=1439
  _globals['_MODELRESPONSE']._serialized_end=1532
  _globals['_GETGLOBALMODELREQUEST']._serialized_start=1534
  _globals['_GETGLOBALMODELREQUEST']._serialized_end=1619
  _globals['_GETGLOBALMODELRESPONSE']._serialized_start=1621
  _globals['_GETGLOBALMODELRESPONSE']._serialized_end=1725
  _globals['_HEARTBEAT']._serialized_start=1727
  _globals['_HEARTBEAT']._serialized_end=1768
  _globals['_CLIENTAVAILABLEMESSAGE']._serialized_start=1770
  _globals['_CLIENTAVAILABLEMESSAGE']._serialized_end=1857
  _globals['_LISTCLIENTSREQUEST']._serialized_start=1859
  _globals['_LISTCLIENTSREQUEST']._serialized_end=1939
  _globals['_CLIENTLIST']._serialized_start=1941
  _globals['_CLIENTLIST']._serialized_end=1983
  _globals['_CLIENT']._serialized_start=1985
  _globals['_CLIENT']._serialized_end=2052
  _globals['_REASSIGNREQUEST']._serialized_start=2054
  _globals['_REASSIGNREQUEST']._serialized_end=2163
  _globals['_RECONNECTREQUEST']._serialized_start=2165
  _globals['_RECONNECTREQUEST']._serialized_end=2264
  _globals['_PARAMETER']._serialized_start=2266
  _globals['_PARAMETER']._serialized_end=2305
  _globals['_CONTROLREQUEST']._serialized_start=2307
  _globals['_CONTROLREQUEST']._serialized_end=2391
  _globals['_CONTROLRESPONSE']._serialized_start=2393
  _globals['_CONTROLRESPONSE']._serialized_end=2463
  _globals['_CONNECTIONREQUEST']._serialized_start=2465
  _globals['_CONNECTIONREQUEST']._serialized_end=2484
  _globals['_CONNECTIONRESPONSE']._serialized_start=2486
  _globals['_CONNECTIONRESPONSE']._serialized_end=2546
  _globals['_PROVIDEDFUNCTIONSREQUEST']._serialized_start=2548
  _globals['_PROVIDEDFUNCTIONSREQUEST']._serialized_end=2597
  _globals['_PROVIDEDFUNCTIONSRESPONSE']._serialized_start=2600
  _globals['_PROVIDEDFUNCTIONSRESPONSE']._serialized_end=2772
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._serialized_start=2715
  _globals['_PROVIDEDFUNCTIONSRESPONSE_AVAILABLEFUNCTIONSENTRY']._serialized_end=2772
  _globals['_CLIENTCONFIGREQUEST']._serialized_start=2774
  _globals['_CLIENTCONFIGREQUEST']._serialized_end=2809
  _globals['_CLIENTCONFIGRESPONSE']._serialized_start=2811
  _globals['_CLIENTCONFIGRESPONSE']._serialized_end=2858
  _globals['_CLIENTSELECTIONREQUEST']._serialized_start=2860
  _globals['_CLIENTSELECTIONREQUEST']._serialized_end=2904
  _globals['_CLIENTSELECTIONRESPONSE']._serialized_start=2906
  _globals['_CLIENTSELECTIONRESPONSE']._serialized_end=2951
  _globals['_CLIENTMETAREQUEST']._serialized_start=2953
  _globals['_CLIENTMETAREQUEST']._serialized_end=3009
  _globals['_CLIENTMETARESPONSE']._serialized_start=3011
  _globals['_CLIENTMETARESPONSE']._serialized_end=3047
  _globals['_STOREMODELREQUEST']._serialized_start=3049
  _globals['_STOREMODELREQUEST']._serialized_end=3094
  _globals['_STOREMODELRESPONSE']._serialized_start=3096
  _globals['_STOREMODELRESPONSE']._serialized_end=3132
  _globals['_AGGREGATIONREQUEST']._serialized_start=3134
  _globals['_AGGREGATIONREQUEST']._serialized_end=3173
  _globals['_AGGREGATIONRESPONSE']._serialized_start=3175
  _globals['_AGGREGATIONRESPONSE']._serialized_end=3210
  _globals['_MODELSERVICE']._serialized_start=3686
  _globals['_MODELSERVICE']._serialized_end=3808
  _globals['_CONTROL']._serialized_start=3811
  _globals['_CONTROL']._serialized_end=4126
  _globals['_REDUCER']._serialized_start=4128
  _globals['_REDUCER']._serialized_end=4214
  _globals['_CONNECTOR']._serialized_start=4217
  _globals['_CONNECTOR']._serialized_end=4644
  _globals['_COMBINER']._serialized_start=4647
  _globals['_COMBINER']._serialized_end=4900
  _globals['_FUNCTIONSERVICE']._serialized_start=4903
  _globals['_FUNCTIONSERVICE']._serialized_end=5395
# @@protoc_insertion_point(module_scope)
