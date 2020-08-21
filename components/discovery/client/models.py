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

    reducer = models.ForeignKey('combiner.Reducer', on_delete=models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)

    status = models.CharField(max_length=2, choices=CLIENT_STATUS, default="R")

    timeout = models.IntegerField(default=30)
    timeout_lost = models.IntegerField(default=3600)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)


from django.db.models import F
from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver


@receiver(pre_save, sender=Client)
def assign_combiner(sender, instance=None, *args, **kwargs):

    from combiner.models import Combiner
    from django.db.models import Count
    cmb = Combiner.objects.all() #annotate(client_count=Count('clients').filter(client_count__lte=F('combinerconfiguration__clients_required')))
    if cmb:
        import random
        selected = random.choices(list(cmb), k=1)
        if selected:
            instance.combiner = selected[0]
            instance.status = 'A'


from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    # override for testing
    if instance.username == 'morgan':
        return

    if created:
        Token.objects.create(user=instance)
