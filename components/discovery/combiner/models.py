from django.contrib.auth.models import User
from django.db import models


class Objective(models.Model):
    name = models.CharField(max_length=512)

    REDUCER_STATUS = [
        ('I', 'Idle'),
        ('A', 'Active'),
        ('R', 'Reset'),
        ('L', 'Lost'),
    ]
    PACKAGE_STATUS = [
        ('N', 'Not set'),
        ('R', 'Ready'),
    ]

    package = models.CharField(max_length=512) #package identifier

    storage = models.ForeignKey('Storage',on_delete=models.DO_NOTHING)


class Storage(models.Model):
    type = models.CharField(default='s3', max_length=512)
    hostname = models.CharField(max_length=512)
    port = models.IntegerField()
    access_key = models.CharField(max_length=512)
    secret_key = models.CharField(max_length=512)
    bucket = models.CharField(null=True, blank=True, default='models', max_length=512)
    secure_mode = models.BooleanField(default=False)


class Combiner(models.Model):
    name = models.CharField(max_length=512)
    host = models.CharField(max_length=512)
    port = models.IntegerField()

    COMBINER_STATUS = [
        ('S', 'Starting'),
        ('R', 'Ready'),
        ('I', 'Instructing'),
        ('C', 'Combining'),
        ('D', 'Decommission'),
        ('L', 'Lost')
    ]

    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)

    status = models.CharField(max_length=2, choices=COMBINER_STATUS, default="S")

    # timeout = models.IntegerField(default=30)
    # timeout_lost = models.IntegerField(default=3600)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    """
    # Combiner private storage settings
    # storage_type = models.CharField(default='s3', max_length=512)
    # storage_hostname = models.CharField(max_length=512)
    # storage_port = models.IntegerField()
    # storage_access_key = models.CharField(max_length=512)
    # storage_secret_key = models.CharField(max_length=512)
    # storage_bucket = models.CharField(null=True, blank=True, default='models', max_length=512)
    # storage_secure_mode = models.BooleanField(default=False)

    # Connection and resource limits for this Combiner
    max_clients = models.IntegerField(default=8)

    # Current combiner-global model
    model_id = models.CharField(max_length=512)

    def __str__(self):
        return str(self.name)


class CombinerConfiguration(models.Model):
     Configuration for a job to be executed by a Combiner. 

    # The associated combiner
    combiner = models.ForeignKey(Combiner, on_delete=models.CASCADE)

    # Algorithm-specific configs
    algorithm = models.CharField(default='fedavg', max_length=512)
    round_timeout = models.IntegerField(default=180)
    rounds = models.IntegerField(default=5)
    clients_required = models.IntegerField(default=1)
    clients_requested = models.IntegerField(default=None)
    ml_framework = models.CharField(default='keras_sequential', max_length=512)
    nr_local_epochs = models.IntegerField(default=1)
    local_batch_size = models.IntegerField(default=32)

    # Model ID to start from
    model_id = models.CharField(max_length=512)
    # seed = models.CharField(max_length=512)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    COMBINER_CONFIG_STATUS = [
        ('R', 'Ready'),
        ('E', 'Executing'),
        ('D', 'Done'),
    ]
    status = models.CharField(max_length=2, choices=COMBINER_CONFIG_STATUS, default="R")
"""