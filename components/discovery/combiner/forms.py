from django import forms

from .models import Combiner, CombinerConfiguration


class CombinerForm(forms.ModelForm):
    class Meta:
        model = Combiner
        fields = ['name', 'status', 'timeout', 'timeout_lost']


class CombinerConfigurationForm(forms.ModelForm):
    class Meta:
        model = CombinerConfiguration
        fields = ['combiner', 'timeout', 'rounds', 'clients_required', 'seed', 'status', 'storage_type',
                  'storage_hostname', 'storage_port', 'storage_access_key', 'storage_secret_key', 'storage_bucket',
                  'storage_secure_mode']
