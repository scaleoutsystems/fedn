from .models import Client

app_name = 'client'


from rest_framework import routers, serializers, viewsets


# Serializers define the API representation.
class ClientSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Client
        fields = ['name', 'status', 'user',]


# ViewSets define the view behavior.
class ClientViewSet(viewsets.ModelViewSet):
    queryset = Client.objects.all()
    serializer_class = ClientSerializer
