from django.contrib.auth.models import User
from django.db import models


class Client(models.Model):
    name = models.CharField(max_length=512)

    CLIENT_STATUS = [
        ('R', 'Ready'),
        ('A', 'Assigned'),
        ('L', 'Lost')
    ]

    combiner = models.ForeignKey('combiner.Combiner', on_delete=models.DO_NOTHING)
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)

    status = models.CharField(max_length=2, choices=CLIENT_STATUS, default="R")

    timeout = models.IntegerField(default=30)
    timeout_lost = models.IntegerField(default=3600)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)


from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)