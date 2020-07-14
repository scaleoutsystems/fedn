import uuid

from models.models import Model
from .models import AllianceInstance
from django.conf import settings

import requests as r
import random
from django.db.models import Q

# NODEPORT REAL STARTRANGE 30000
NODEPORT_RANGE_START = 30100
NODEPORT_RANGE_END = 32000


# NODEPORT REAL ENDRANGE 32767


def get_free_port(reserved=None):
    for _ in range(10):
        port = random.randint(NODEPORT_RANGE_START, NODEPORT_RANGE_END)
        qs = AllianceInstance.objects.filter(Q(controller_port=port) | Q(minio_port=port))
        if reserved:
            if port == reserved:
                continue
        if len(qs) == 0:
            return port
    return None

def create_alliance_from_model(user, project, model_uid):
    uid = uuid.uuid4()
    uid = 'a' + str(uid)

    ai = AllianceInstance()
    ai.uid = uid
    ai.controller_url = 'http://' + str(uid) + '-controller'
    ai.orchestrator_url = 'http://' + str(uid) + '-combiner'
    ai.controller_port = get_free_port()
    # NOTE BY AUTHOR: in the land of huge caches and no-mechanical disks and forced write-throughs to ensure entropy:
    # by murhpy law whatever can happen will happen so better make sure no port can be selected random twice!
    ai.minio_port = get_free_port(ai.controller_port)
    ai.orchestrator_status = AllianceInstance.ORCHESTRATOR_STATE[1][0]

    ai.project = project


    ai.seed_model = Model.objects.filter(uid=model_uid).first()

    import os
    if "TELEPRESENCE_ROOT" in os.environ:
        domain = '3.231.229.94.nip.io'
    else:
        domain = settings.DOMAIN

    parameters = {'release': str(ai.uid),
                  'chart': 'alliance',
                  'service.minio': 443,
                  'service.controller': str(ai.controller_port),
                  'minio.access_key': str(project.project_key),
                  'minio.secret_key': str(project.project_secret),
                  'model': str(model_uid),
                  'user': str(user),
                  'alliance.project': str(project.slug),
                  'global.domain': str(domain),
                  'alliance.apiUrl': settings.STUDIO_URL}

    url = settings.CHART_CONTROLLER_URL + '/deploy'

    print("MAKING REQUEST TO: {} with parameters {}".format(url, parameters))

    retval = r.get(url, parameters)
    print("CREATE_ALLIANCE:helm chart creator returned {}".format(retval))

    if 200 <= retval.status_code < 205:
        ai.save()
        return ai.uid

    return None


def create_alliance(user, project):
    uid = uuid.uuid4()
    uid = 'a' + str(uid)

    ai = AllianceInstance()
    ai.uid = uid
    ai.controller_url = 'http://' + str(uid) + '-controller'
    ai.orchestrator_url = 'http://' + str(uid) + '-combiner'
    ai.controller_port = get_free_port()
    # NOTE BY AUTHOR: in the land of huge caches and no-mechanical disks and forced write-throughs to ensure entropy:
    # by murhpy law whatever can happen will happen so better make sure no port can be selected random twice!
    ai.minio_port = get_free_port(ai.controller_port)


    ai.project = project

    ai.seed_model = Model.objects.filter(project=project).order_by('id').first()

    parameters = {'release': str(ai.uid),
                  'chart': 'alliance',
                  'minio_port': str(ai.minio_port),
                  'controller_port': str(ai.controller_port),
                  'user': str(user),
                  'sleep': 90,
                  'rounds': 10,
                  'project': str(project.slug),
                  'alliance.apiUrl': settings.STUDIO_URL }

    url = settings.CHART_CONTROLLER_URL + '/deploy'

    print("MAKING REQUEST TO: {} with parameters {}".format(url,parameters))

    retval = r.get(url, parameters)
    print("CREATE_ALLIANCE:helm chart creator returned {}".format(retval))

    if 200 <= retval.status_code < 205:
        ai.save()
        return ai.uid

    return None


def destroy_alliance(uid):
    ai = AllianceInstance.objects.filter(uid=uid).first()
    if ai:
        retval = r.get(settings.CHART_CONTROLLER_URL + '/delete/?release={}'.format(str(uid)))

        if retval:
            ai.delete()
            return True

    return False


def start_alliance(ai, rounds):
    if ai:
        retval = r.get(ai.orchestrator_url + '')

        if retval.json()['combiner']['status']:
            print("already started, resetting")
            #TODO move reset into this clause after figuring out why json retval conversion is not working!

        r.get(ai.orchestrator_url + '/reset')


        parameters = {
            'model': ai.seed_model,
            'rounds': rounds
        }
        retval = r.get(ai.orchestrator_url + '/start',parameters)
        if retval.status_code >= 200 or retval.status_code < 205:
            ai.orchestrator_status = AllianceInstance.ORCHESTRATOR_STATE[1][0]
            ai.save()
            return True

    return False


def stop_alliance(ai):
    if ai:
        retval = r.get(ai.orchestrator_url + '/stop')
        if retval.status_code >= 200 or retval.status_code < 205:
            ai.orchestrator_status = AllianceInstance.ORCHESTRATOR_STATE[0][0]
            ai.save()
            return True

    return False
