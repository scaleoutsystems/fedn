from django.core.validators import RegexValidator
from django.db import models


class AllianceInstance(models.Model):
    uid = models.CharField(max_length=512)
    controller_url = models.CharField(
        validators=[RegexValidator(regex='(http[s]?://)([a-zA-Z|0-9|-]+)([.][a-zA-Z]+)?', message='Not a valid URL')],
        max_length=512)
    orchestrator_url = models.CharField(
        validators=[RegexValidator(regex='(http[s]?://)([a-zA-Z|0-9|-]+)([.][a-zA-Z]+)?', message='Not a valid URL')],
        max_length=512)

    ORCHESTRATOR_STATE = (
        ('unconfigured', 'unconfigured'),
        ('stopped', 'stopped'),
        ('running', 'running')
    )
    orchestrator_status = models.CharField(choices=ORCHESTRATOR_STATE, max_length=32, default=ORCHESTRATOR_STATE[0][0])
    controller_port = models.IntegerField()
    minio_port = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    project = models.ForeignKey('projects.Project', on_delete=models.DO_NOTHING, related_name='instance_project')
    seed_model = models.ForeignKey('models.Model', on_delete=models.DO_NOTHING, related_name='seed_model', blank=True, null=True)

    ORCHESTRATOR_ALGORITHM = (
        ('Federated Averaging', 'Federated Averaging'),
        ('Custom', 'Custom')
    )
    orchestrator_algorithm = models.CharField(choices=ORCHESTRATOR_ALGORITHM, max_length=32,
                                              default=ORCHESTRATOR_ALGORITHM[0][0])


class Event(models.Model):
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    alliance = models.ForeignKey('alliance_admin.AllianceInstance', on_delete=models.DO_NOTHING, related_name='alliance_events')
