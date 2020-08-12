from django.urls import path

from . import views

app_name = 'combiners'

urlpatterns = [
    path('', views.index, name='index'),
    path('combiners/', views.combiner, name='combiner'),
    path('combiners/<combiner>/start', views.start, name='start'),
    path('combiners/<combiner>/stop', views.stop, name='stop'),
    path('combiners/<combiner>/snapshot', views.snapshot, name='snapshot'),
    path('combiners/<combiner>/configure', views.configure, name='configure'),
    path('combiners/<combiner>/', views.details, name='details'),
]
