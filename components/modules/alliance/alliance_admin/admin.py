from django.contrib import admin

from .models import AllianceInstance, Event
# Register your models here.
admin.site.register(AllianceInstance)
admin.site.register(Event)
