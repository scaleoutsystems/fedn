from django.shortcuts import render

from .models import Combiner, CombinerConfiguration
from client.models import Client


# Create your views here.
def index(request):
    template = 'index.html'

    try:
        combiners = Combiner.objects.all()
    except TypeError as err:
        combiners = []
        print(err)

    # request.session['next'] = '/combiners/'
    return render(request, template, locals())


def combiner(request):
    template = 'combiner.html'

    try:
        combiners = Combiner.objects.all()
    except TypeError as err:
        combiners = []
        print(err)

    # request.session['next'] = '/combiners/'
    return render(request, template, locals())


# Create your views here.
def configure(request, combiner):
    template = 'configure.html'

    try:
        combiners = Combiner.objects.filter(name=combiner).first()
    except TypeError as err:
        combiners = []
        print(err)

    # request.session['next'] = '/combiners/'
    return render(request, template, locals())


def start(request, combiner):
    next_page= '/controller/combiners/{}'.format(combiner)
    template = 'details.html'

    try:
        combiner = Combiner.objects.filter(name=combiner).first()
    except TypeError as err:
        combiner = None
        print(err)

    combiner.status = 'I'
    combiner.save()
    clients = Client.objects.filter(combiner=combiner).first()

    configuration = CombinerConfiguration.objects.filter(combiner=combiner).first()

    from django.http import HttpResponseRedirect
    return HttpResponseRedirect(next_page)

def stop(request, combiner):
    next_page = '/controller/combiners/{}'.format(combiner)
    template = 'details.html'

    try:
        combiner = Combiner.objects.filter(name=combiner).first()
    except TypeError as err:
        combiner = None
        print(err)

    combiner.status = 'S'
    combiner.save()
    clients = Client.objects.filter(combiner=combiner).first()

    configuration = CombinerConfiguration.objects.filter(combiner=combiner).first()

    #return render(request, template, locals())
    from django.http import HttpResponseRedirect
    return HttpResponseRedirect(next_page)


def snapshot(request, combiner):
    template = 'combiner.html'

    request.session['next'] = '/combiners/{}'.format(combiner)
    return render(request, template, locals())


# Create your views here.
def control(request, combiner):
    template = 'control.html'
    from .forms import CombinerForm

    if request.method == 'POST':
        form = CombinerForm(request.POST)
        if form.is_valid():
            obj, created = Combiner.objects.update_or_create(form.cleaned_data)
            # obj = form.save()
            # obj.save()
            if created:
                print("a new combiner was created!", flush=True)
            else:
                print("combiner was configured", flush=True)

    try:
        combiner = Combiner.objects.filter(name=combiner).first()
    except TypeError as err:
        combiner = None
        print(err)

    form = CombinerForm(instance=combiner)

    # request.session['next'] = '/combiners/'
    return render(request, template, locals())


def details(request, combiner):
    template = 'details.html'

    stat = None
    try:
        combiner = Combiner.objects.filter(name=combiner).first()
        stat = combiner.status
    except TypeError as err:
        combiner = None
        print(err)

    clients = Client.objects.filter(combiner__name=combiner)

    if stat =='I':
        status = 'INSTRUCTING'
    if stat == 'S':
        status = 'READY'
    if stat == 'R':
        status = 'READY'
    if stat == 'C':
        status = 'COMBINING'

    configuration = CombinerConfiguration.objects.filter(combiner=combiner).first()

    return render(request, template, locals())
