from django.db import models

from django.contrib.auth.models import User
from django.db import models


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

    timeout = models.IntegerField(default=30)
    timeout_lost = models.IntegerField(default=3600)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)


class CombinerConfiguration(models.Model):

    combiner = models.ForeignKey(Combiner, on_delete=models.CASCADE)
    timeout = models.IntegerField(default=180)
    rounds = models.IntegerField(default=5)
    clients_required = models.IntegerField(default=2)
    seed = models.CharField(max_length=512)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    COMBINER_CONFIG_STATUS = [
        ('R', 'Ready'),
        ('E', 'Executing'),
        ('D', 'Done'),
    ]
    status = models.CharField(max_length=2, choices=COMBINER_CONFIG_STATUS, default="R")

