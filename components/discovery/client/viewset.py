from .models import Client

app_name = 'client'


from rest_framework import routers, serializers, viewsets


# Serializers define the API representation.
class ClientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Client
        fields = ['name', 'status', 'user','user_id','combiner','timeout','timeout_lost']


# ViewSets define the view behavior.
class ClientViewSet(viewsets.ModelViewSet):
    queryset = Client.objects.all()
    serializer_class = ClientSerializer
    lookup_field = 'name'
