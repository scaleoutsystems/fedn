from django.contrib.auth.models import User
from django.db import models


class Client(models.Model):
    name = models.CharField(max_length=512)

    CLIENT_STATUS = [
        ('R', 'Ready'),
        ('A', 'Assigned'),
        ('E', 'Reassign'),
        ('L', 'Lost')
    ]

    combiner = models.ForeignKey('combiner.Combiner', on_delete=models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)

    status = models.CharField(max_length=2, choices=CLIENT_STATUS, default="R")

    timeout = models.IntegerField(default=30)
    timeout_lost = models.IntegerField(default=3600)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)


from django.db.models import F
from django.db.models.signals import post_save
from django.dispatch import receiver


#@receiver(post_save, sender=Client)
#def create_client_configuration(sender, instance=None, created=False, **kwargs):
#    if created:
#
#        from combiner.models import Combiner
#        combiners = Combiner.objects.filter(clients__lt=F('combinerconfiguration__clients_required'))
#        if combiners:
#            import random
#           selected = random.choices(list(combiners))
#            if selected:
#                instance.combiner = selected


#class ClientConfiguration(models.Model):
#    client = models.ForeignKey('client.Client', on_delete=models.DO_NOTHING)

#    storage_type = models.CharField(default='s3', max_length=512)
#    storage_hostname = models.CharField(max_length=512)
#    storage_port = models.IntegerField()
#    storage_access_key = models.CharField(max_length=512)
#    storage_secret_key = models.CharField(max_length=512)
#    storage_bucket = models.CharField(null=True, blank=True, default='models', max_length=512)
#    storage_secure_mode = models.BooleanField(default=False)

# from django.conf import settings
# from django.db.models.signals import post_save
# from django.dispatch import receiver
# from rest_framework.authtoken.models import Token

# @receiver(post_save, sender=settings.AUTH_USER_MODEL)
# def create_auth_token(sender, instance=None, created=False, **kwargs):
#    if created:
#        Token.objects.create(user=instance)
