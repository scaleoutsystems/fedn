from django.urls import path
from . import views

app_name = 'alliance_admin'

urlpatterns = [
    path('projects/<user>/<project>/alliance_admin/', views.index, name='index'),
    path('projects/<user>/<project>/alliance_admin/<str:uid>/logs', views.logs, name='logs'),
    path('projects/<user>/<project>/alliance_admin/create', views.create, name='create'),
    path('projects/<user>/<project>/alliance_admin/<str:uid>/details', views.details, name='details'),
    path('projects/<user>/<project>/alliance_admin/<int:id>/start', views.start, name='start'),
    path('projects/<user>/<project>/alliance_admin/<int:id>/stop', views.stop, name='stop'),
    path('projects/<user>/<project>/alliance_admin/<str:uid>/project', views.project, name='project'),
    path('projects/<user>/<project>/alliance_admin/<str:uid>/log', views.log, name='log'),

]
