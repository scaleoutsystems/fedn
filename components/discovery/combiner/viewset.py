from .models import Combiner

app_name = 'combiner'


from rest_framework import routers, serializers, viewsets


# Serializers define the API representation.
class CombinerSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Combiner
        fields = ['name', 'status', 'user',]


# ViewSets define the view behavior.
class CombinerViewSet(viewsets.ModelViewSet):
    queryset = Combiner.objects.all()
    serializer_class = CombinerSerializer
