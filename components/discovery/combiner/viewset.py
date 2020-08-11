from .models import Combiner, CombinerConfiguration

app_name = 'combiner'

from rest_framework import serializers, viewsets


# Serializers define the API representation.
class CombinerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Combiner
        fields = ['name', 'status', 'user', 'host', 'port','storage_type','storage_hostname', 'storage_port', 'storage_access_key', 'storage_secret_key',
                  'storage_bucket', 'storage_secure_mode','max_clients','model_id']


# ViewSets define the view behavior.
class CombinerViewSet(viewsets.ModelViewSet):
    queryset = Combiner.objects.all()
    serializer_class = CombinerSerializer
    lookup_field = 'name'


class CombinerConfigurationSerializer(serializers.ModelSerializer):
    class Meta:
        model = CombinerConfiguration
        fields = ['combiner','algorithm','ml_framework','round_timeout', 'rounds', 'clients_required','clients_requested', 
                  'nr_local_epochs','local_batch_size','model_id', 'updated_at', 'created_at', 'status']


# ViewSets define the view behavior.
class CombinerConfigurationViewSet(viewsets.ModelViewSet):
    queryset = CombinerConfiguration.objects.all()
    serializer_class = CombinerConfigurationSerializer
    lookup_field = 'combiner__name'
    filter_fields = ['combiner__name', 'status']
