from rest_framework.serializers import ModelSerializer

from .models import ResponseEvent, StatusEvent, RequestEvent, AllianceInstance


class AllianceInstanceSerializer(ModelSerializer):
    class Meta:
        model = AllianceInstance
        fields = ('uid', 'controller_url', 'orchestrator_url', 'orchestrator_status', 'controller_port', 'minio_port',
                  'project', 'seed_model', 'orchestrator_algorithm')


class ResponseEventSerializer(ModelSerializer):
    class Meta:
        model = ResponseEvent
        fields = ('response', 'created_at')


class StatusEventSerializer(ModelSerializer):
    class Meta:
        model = StatusEvent
        fields = ('client', 'status', 'created_at')


class RequestEventSerializer(ModelSerializer):
    class Meta:
        model = RequestEvent
        fields = ('client', 'command', 'payload', 'extra', 'created_at', 'alliance_uid')
