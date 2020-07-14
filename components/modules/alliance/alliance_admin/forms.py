from django import forms
from django.core.validators import RegexValidator


class AllianceInstanceForm(forms.Form):
    uid = forms.CharField(max_length=521)
    controller_url = forms.CharField(
        validators=[RegexValidator(regex='(http[s]?://)([a-zA-Z|0-9|-]+)([.][a-zA-Z]+)?', message='Not a valid URL')],
        max_length=512
    )
    orchestrator_url = forms.CharField(
        validators=[RegexValidator(regex='(http[s]?://)([a-zA-Z|0-9|-]+)([.][a-zA-Z]+)?', message='Not a valid URL')],
        max_length=512
    )
    orchestrator_status = forms.CharField(max_length=32)
    controller_port = forms.IntegerField()
    minio_port = forms.IntegerField()
    project = forms.CharField()
    seed_model = forms.CharField()
    orchestrator_algorithm = forms.CharField()

