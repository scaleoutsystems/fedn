from django.contrib import admin

# Register your models here.
from .models import Combiner, CombinerConfiguration

admin.site.register(Combiner)
admin.site.register(CombinerConfiguration)
