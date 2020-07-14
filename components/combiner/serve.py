import os
from flask import Flask
from flask import request
import json
import threading
import subprocess

app = Flask(__name__)
app.orch_status = False


def run(seed_model, rounds=5):
    p = subprocess.Popen('fedn run fedavg -s {} -r {}'.format(seed_model, rounds).split(), shell=False,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()


@app.route('/')
def index():
    return json.dumps({'combiner': {'status': app.orch_status}})


@app.route('/start')
def start():

    seed_model = request.args.get('model','')
    rounds = request.args.get('rounds','5')
    if app.orch_status:
        return json.dumps({'combiner': {'status': app.orch_status}})
    else:
        app.orch_status = True

        threading.Thread(target=run, args=(seed_model, rounds), daemon = True).start()
        # subprocess.run(["fedn","run","fedavg","-s","mnist-seed-model.h5","-r","5"])

        return json.dumps({'combiner': {'status': 'starting'}})

    return json.dumps({'combiner': {'status': 'Error'}})


@app.route('/reset')
def reset():
    app.orch_status = False
    return json.dumps({'combiner': {'status': 'reset OK'}})
