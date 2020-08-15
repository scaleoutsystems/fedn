from flask import Flask, jsonify, render_template
import pymongo
from bson.objectid import ObjectId
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy
import plotly.express as px
import plotly.figure_factory as ff
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime,timedelta

import os

app = Flask(__name__)

c = pymongo.MongoClient()
mc = pymongo.MongoClient('mongo',27017,username=os.environ['MDBUSR'],password=os.environ['MDBPWD'])
mdb = mc[os.environ['ALLIANCE_UID']]
alliance = mdb["status"]

@app.route('/')
def hello_world():
    return 'Welcome to the Scaleout Systems FEDn dashboard:'

@app.route("/table", methods=['POST', 'GET'])
def table():

    metrics = alliance.find_one({'type': 'MODEL_VALIDATION'})
    if metrics == None:
        fig = go.Figure(data=[])
        fig.update_layout(title_text='No data currently available')
        pio.write_html(fig, file= 'templates/table.html')
        return render_template('table.html')

    data = json.loads(metrics['data'])
    data = json.loads(data['data'])
    valid_metrics = []
    for metric,val in data.items():
        # Check if scalar - is this robust ? 
        if isinstance(val,float):
            valid_metrics.append(metric)

    all_vals=[]
    models = []
    for metric in valid_metrics:
        validations = {}
        for post in alliance.find({'type': 'MODEL_VALIDATION'}):
            e = json.loads(post['data'])
            try:
                validations[e['modelId']].append(json.loads(e['data'])[metric])
            except KeyError:
                validations[e['modelId']] = [json.loads(e['data'])[metric]]

        vals = []
        models =[]
        for model,data in validations.items():
            vals.append(numpy.mean(data))
            models.append(model)
        all_vals.append(vals)

    header_vals = valid_metrics
    all_vals[0].reverse()
    all_vals[1].reverse()

    fig = go.Figure(data=[go.Table(
    header=dict(values=['Model ID']+header_vals,
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),

    cells=dict(values=[models,       
                        all_vals[0], # 1st column
                        all_vals[1]],# 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
    ])

    fig.update_layout(width=800, height=600)
    fig.update_layout(title_text='Summary: mean metrics')
    pio.write_html(fig, file= 'templates/table.html')
    
    return render_template('table.html')


@app.route("/timeline", methods=['POST', 'GET'])
def timeline():

    trace_data = []
    x = []
    y = []
    base = []
    for p in alliance.find({'type': 'MODEL_UPDATE_REQUEST'}):
        e = json.loads(p['data'])
        cid = e['correlationId']
        for cc in alliance.find({'sender':p['sender'],'type':'MODEL_UPDATE'}):
            da = json.loads(cc['data'])
            if da['correlationId'] == cid:
                cp = cc

        cd = json.loads(cp['data'])
        tr = datetime.strptime(e['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        tu = datetime.strptime(cd['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        ts = tu-tr
        base.append(tr.timestamp())
        x.append(ts.total_seconds())
        y.append(p['sender']['name'])

    trace_data.append(go.Bar(
        x=x,
        y=y,
        orientation='h',
        base = base,
        marker=dict(color='royalblue'),
        name="Training",
    ))

    x = []
    y = []
    base = []
    for p in alliance.find({'type': 'MODEL_VALIDATION_REQUEST'}):
        e = json.loads(p['data'])
        cid = e['correlationId']
        for cc in alliance.find({'sender':p['sender'],'type':'MODEL_VALIDATION'}):
            da = json.loads(cc['data'])
            if da['correlationId'] == cid:
                cp = cc
        cd = json.loads(cp['data'])
        tr = datetime.strptime(e['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        tu = datetime.strptime(cd['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        ts = tu-tr
        base.append(tr.timestamp())
        x.append(ts.total_seconds())
        y.append(p['sender']['name'])

    trace_data.append(go.Bar(
        x=x,
        y=y,
        orientation='h',
        base = base,
        marker=dict(color='lightskyblue'),
        name="Validation", 
    ))

    layout = go.Layout(
        barmode='stack',
        showlegend=True,
    )

    fig = go.Figure(data=trace_data,layout=layout)

    fig.update_xaxes(title_text='Timestamp')
    fig.update_layout(title_text='Alliance timeline')

    tab = go.Figure(data=[go.Table(
    header=dict(values=['Model updates', 'Model Validations'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[[100, 90, 80, 90], # 1st column
                       [95, 85, 75, 95]], # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
    ])

    tab.update_layout(width=500, height=300)
    tab.update_layout(title_text='Summary')


    with open('templates/clients.html','w') as fh:
        fh.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    return render_template('clients.html')

@app.route("/ml", methods=['POST', 'GET'])
def ml():

    metrics = alliance.find_one({'type': 'MODEL_VALIDATION'})
    if metrics == None:
        fig = go.Figure(data=[])
        fig.update_layout(title_text='No data currently available')
        pio.write_html(fig, file= 'templates/table.html')
        return render_template('table.html')
    
    data = json.loads(metrics['data'])
    data = json.loads(data['data'])
    valid_metrics = []
    for metric,val in data.items():
        # Check if scalar - is this robust ? 
        if isinstance(val,float):
            valid_metrics.append(metric)
 
    # Assemble a dict with all validations
    validations = {}
    clients = {}
 #   for post in alliance.find({'type': 'MODEL_VALIDATION'}):
 #       e = json.loads(post['data'])
 #      clients[post['sender']['name']]=[]

    for post in alliance.find({'type': 'MODEL_VALIDATION'}):
        try:
            e = json.loads(post['data'])
            clients[post['sender']['name']].append(json.loads(e['data'])[metric])
        except KeyError:
            clients[post['sender']['name']]=[]

    rounds = []
    traces_data = []

    for c in clients:
        print(clients[c],flush=True)
        traces_data.append(go.Scatter(
            x=rounds,
            y=clients[c],
            name=c 
        )) 
    fig = go.Figure(traces_data)
    fig.update_xaxes(title_text='Rounds')
    fig.update_yaxes(title_text='MAE', tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.update_layout(title_text='Mean Absolute Error Plot')
    pio.write_html(fig, file= 'templates/basic-line.html')
    
    return render_template('basic-line.html')

@app.route("/box", methods=['POST', 'GET'])
def box():

    #metric = 'mae'
    metric = 'accuracy'

    metrics = alliance.find_one({'type': 'MODEL_VALIDATION'})
    if metrics == None:
        fig = go.Figure(data=[])
        fig.update_layout(title_text='No data currently available')
        pio.write_html(fig, file= 'templates/table.html')
        return render_template('table.html')


    data = json.loads(metrics['data'])
    data = json.loads(data['data'])
    valid_metrics = []
    for metric,val in data.items():
        # Check if scalar - is this robust ? 
        if isinstance(val,float):
            valid_metrics.append(metric)

    print(valid_metrics,flush=True)
    metric = 'accuracy'

    validations = {}
    for post in alliance.find({'type': 'MODEL_VALIDATION'}):
        e = json.loads(post['data'])
        try:
            validations[e['modelId']].append(json.loads(e['data'])[metric])
        except KeyError:
            validations[e['modelId']] = [json.loads(e['data'])[metric]]

    box = go.Figure()

    x = []
    y = []
    box_trace = []
    for model_id, acc in validations.items():
        x.append(model_id)
        y.append(numpy.mean([float(i) for i in acc]))
        if len(acc) >= 2:
            box.add_trace(go.Box(y=acc,name=str(model_id),marker_color="royalblue",showlegend=False))

    rounds = list(range(len(y)))
    box.add_trace(go.Scatter(
        x=x,
        y=y,
        name='Mean'
    ))

    box.update_xaxes(title_text='Model ID')
    box.update_yaxes(title_text='MAE', tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    box.update_layout(title_text='Mean Absolute Error')
    pio.write_html(box, file= 'templates/basic-box.html')
    
    return render_template('basic-box.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5111)

