from .models import Combiner, CombinerConfiguration

app_name = 'combiner'


from rest_framework import routers, serializers, viewsets


# Serializers define the API representation.
class CombinerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Combiner
        fields = ['name', 'status', 'user','host','port']


# ViewSets define the view behavior.
class CombinerViewSet(viewsets.ModelViewSet):
    queryset = Combiner.objects.all()
    serializer_class = CombinerSerializer
    lookup_field = 'name'


class CombinerConfigurationSerializer(serializers.ModelSerializer):
    class Meta:
        model = CombinerConfiguration
        fields = ['combiner', 'timeout','rounds','clients_required','seed', 'updated_at','created_at', 'status', ]


# ViewSets define the view behavior.
class CombinerConfigurationViewSet(viewsets.ModelViewSet):
    queryset = CombinerConfiguration.objects.all()
    serializer_class = CombinerConfigurationSerializer
    lookup_field = 'combiner__name'

