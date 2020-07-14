import requests
from minio import Minio, ResponseError
from datetime import timedelta

import studio.settings as settings

from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse

from projects.models import Project
from models.models import Model
from .models import AllianceInstance, Event
from .helpers import create_alliance, create_alliance_from_model, start_alliance, stop_alliance
from .dtos import ModelDto
from .forms import AllianceInstanceForm

def index(request, user, project):
    template = 'alliance_admin.html'
    project = Project.objects.filter(slug=project).first()

    alliance_instances = AllianceInstance.objects.filter(project=project)
    return render(request, template, locals())


def logs(request, user, project, uid):
    template = 'event_log.html'

    project = Project.objects.filter(slug=project).first()
    requests = RequestEvent.objects.filter(alliance_uid=uid).order_by('-created_at')

    return render(request, template, locals())


def create(request, user, project):
    project = Project.objects.filter(slug=project).first()
    models = Model.objects.all()
    template = 'alliance_create.html'

    if request.method == 'POST':

        model_uid = request.POST.get('model', None)
        # timeout = request.POST.get('timeout', 90)
        # rounds = request.POST.get('rounds', 10)
        if model_uid:
            # form = AllianceInstanceForm(request.POST)

            # if form.is_valid():
            #    request_body = {
            #        'uid': form.cleaned_data['uid'],
            #        'controller_url': form.cleaned_data['controller_url'],
            #        'orchestrator_url': form.cleaned_data['orchestrator_url'],
            #        'orchestrator_status': form.cleaned_data['orchestrator_status'],
            #        'controller_port': form.cleaned_data['controller_port'],
            #        'minio_port': form.cleaned_data['minio_port'],
            #        'project': form.cleaned_data['project'],
            #        'seed_model': form.cleaned_data['seed_model'],
            #        'orchestrator_algorithm': form.cleaned_data['orchestrator_algorithm']
            #    }

            ai = create_alliance_from_model(user=user, project=project, model_uid=model_uid)
            if ai:
                print("creating alliance resources successful. Saving instance!")

                # requests.put('{}/projects/{}/{}/alliance_admin/alliance/{}/'.format(settings.DOMAIN, user,
                #                                                                   project.slug, ai.pk), data=request_body)

                url = '/projects/{}/{}/alliance_admin/{}/details'.format(request.user, project.slug, str(ai))
            else:
                url = '/projects/{}/{}/alliance_admin/'.format(request.user, project.slug)
        else:
            url = '/projects/{}/{}/alliance_admin/'.format(request.user, project.slug)

        return HttpResponseRedirect(url)
    else:
        form = AllianceInstanceForm()

    return render(request, template, locals())

    next_page = '/projects/{user}/{project}/alliance_admin/'.format(user=user, project=project.slug)
    if retval:
        next_page = '/projects/{user}/{project}/alliance_admin/{id}/details'.format(user=user,
                                                                                    project=project.slug,
                                                                                    id=str(retval))

    return HttpResponseRedirect(next_page)


def details(request, user, project, uid):
    template = 'alliance_details.html'

    project = Project.objects.filter(slug=project).first()
    ai = AllianceInstance.objects.filter(uid=uid).first()
    models = Model.objects.filter(project=project)
    events  = Event.objects.filter(alliance=ai)
    checkpoints = {}
    try:
        r = requests.get("http://{}-checkpointer/".format(ai.uid))

        checkpoints.update(r.json())
    except Exception as e:
        pass

    import os
    if "TELEPRESENCE_ROOT" in os.environ:
        domain = '3.231.229.94.nip.io'
    else:
        domain = settings.DOMAIN

    minio_url = "{}.minio.{}:443".format(project.slug, domain)
    # minio_client = Minio(minio_url, access_key=project.project_key, secret_key=project.project_secret, secure=False)
    from urllib3.poolmanager import PoolManager
    manager = PoolManager(10, cert_reqs='CERT_NONE', assert_hostname=False)
    minio_client = Minio(minio_url,
                         access_key=project.project_key,
                         secret_key=project.project_secret,
                         secure=True, http_client=manager)
    try:
        objects = minio_client.list_objects_v2('alliance', prefix='', recursive=True, start_after='')

        if objects:
            dtos = []
            for obj in objects:
                downloadable_link = minio_client.presigned_get_object('alliance', obj.object_name,
                                                                      expires=timedelta(days=7))

                dto = ModelDto(obj.bucket_name, obj.object_name.encode('utf-8'), obj.last_modified, obj.etag, obj.size,
                               obj.content_type, downloadable_link)

                dtos.append(dto)
    except Exception as e:
        print(e)

    """if request.method == 'POST':
        form = AllianceInstanceForm(request.POST)

        if form.is_valid():
            request_body = {
                'uid': form.cleaned_data['uid'],
                'controller_url': form.cleaned_data['controller_url'],
                'orchestrator_url': form.cleaned_data['orchestrator_url'],
                'orchestrator_status': form.cleaned_data['orchestrator_status'],
                'controller_port': form.cleaned_data['controller_port'],
                'minio_port': form.cleaned_data['minio_port'],
                'project': form.cleaned_data['project'],
                'seed_model': form.cleaned_data['seed_model'],
                'orchestrator_algorithm': form.cleaned_data['orchestrator_algorithm']
            }

            requests.put('{}/projects/{}/{}/alliance_admin/alliance/{}/'.format(settings.DOMAIN, user,
                                                                                project.slug, ai.pk), data=request_body)

            url = '/projects/{}/{}/alliance_admin/{}/details'.format(request.user, project.slug, str(ai.uid))
        else:
            url = '/projects/{}/{}/alliance_admin/'.format(request.user, project.slug)

        return HttpResponseRedirect(url)
    else:
        form = AllianceInstanceForm()
    """
    return render(request, template, locals())


def start(request, user, project, id):
    template = 'alliance_details.html'
    project = Project.objects.filter(slug=project).first()

    if request.method == "POST":
        rounds = request.POST.get('rounds', '5')

        ai = AllianceInstance.objects.filter(id=id).first()
        start_alliance(ai, rounds)

    return render(request, template, locals())


def stop(request, user, project, id):
    template = 'alliance_details.html'
    project = Project.objects.filter(slug=project).first()

    ai = AllianceInstance.objects.filter(id=id).first()
    stop_alliance(ai)

    return render(request, template, locals())


def project(request, user, project, uid):
    ctx = """auth_url: http://{domain}/api/api-token-auth
username: {name}
access_key: {name}
password: {secret_key}
so_domain_name: {domain}

Project:
  project_name: TestProject
  project_id:

Alliance:
  alliance_name: testalliance
  controller_host: {domain}
  controller_port: {port}

  Repository:
    minio_host: {project}-minio.{domain}
    minio_port: 443
    minio_bucket: alliance
    minio_access_key: {access_key}
    minio_secret_key: {secret_key}
    minio_secure_mode: True

  Member:
    name: {name}
    entry_points:
      predict:
        command: python3 predict.py
      train:
        command: python3 train.py
      validate:
        command: python3 validate.py

    """
    project = Project.objects.filter(slug=project).first()
    ai = AllianceInstance.objects.filter(uid=uid).first()

    import string
    import random
    letters = string.ascii_lowercase
    name = ''.join(random.choice(letters) for i in range(16))

    response = HttpResponse(content=ctx.format(domain=settings.DOMAIN, port=ai.controller_port, project=project.slug,
                                               access_key=project.project_key, secret_key=project.project_secret,name=name))


    response['Content-Type'] = 'application/x-yaml'
    return response

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def log(request, user, project, uid):

    ai = AllianceInstance.objects.filter(uid=uid).first()
    status = 'Error'
    if request.method == 'POST':
        message = request.POST.get('message',None)
        if message:
            event = Event()
            event.text = message
            event.alliance = ai
            event.save()
            status = 'OK'

    response = {
        'status': status,
    }
    return JsonResponse(response)