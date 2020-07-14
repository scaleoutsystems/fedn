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

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Welcome to the Scaleout Systems FEDn dashboard:'

@app.route("/table", methods=['POST', 'GET'])
def table():
 
  
    c = pymongo.MongoClient()
    mc = pymongo.MongoClient('mongo',27017,username='root',password='example')
    mdb = mc["ac435faef-c2df-442e-b349-7f633d3d5523"]
    alliance = mdb["status"]

    metric = 'mae'
    updates = {}
    for p in alliance.find({'type': 'MODEL_UPDATE'}):
        e = json.loads(p['data'])
        try: 
            updates[p['client']].append(datetime.strptime(p['timestamp'],'%Y-%m-%d %H:%M:%S.%f'))
        except KeyError:
            updates[p['client']] = [datetime.strptime(p['timestamp'],'%Y-%m-%d %H:%M:%S.%f')]

    trace_data = []
    for client,data in updates.items():

        trace_data

    fig = go.Figure(data=[go.Table(
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

    fig.update_layout(width=500, height=300)
    fig.update_layout(title_text='Summary')
    pio.write_html(fig, file= 'templates/table.html')
    
    return render_template('table.html')


@app.route("/clients", methods=['POST', 'GET'])
def clients():
 
  
    c = pymongo.MongoClient()
    mc = pymongo.MongoClient('mongo',27017,username='root',password='example')
    mdb = mc["ac435faef-c2df-442e-b349-7f633d3d5523"]
    alliance = mdb["status"]

    trace_data = []
    x = []
    y = []
    base = []
    for p in alliance.find({'type': 'MODEL_UPDATE_REQUEST'}):
        e = json.loads(p['data'])
        cid = e['correlationId']
        for cc in alliance.find({'client':p['client'],'type':'MODEL_UPDATE'}):
            da = json.loads(cc['data'])
            if da['correlationId'] == cid:
                cp = cc

        cd = json.loads(cp['data'])
        tr = datetime.strptime(e['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        tu = datetime.strptime(cd['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        ts = tu-tr
        base.append(tr.timestamp())
        x.append(ts.total_seconds())
        y.append(p['client'])

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
        for cc in alliance.find({'client':p['client'],'type':'MODEL_VALIDATION'}):
            da = json.loads(cc['data'])
            if da['correlationId'] == cid:
                cp = cc
        cd = json.loads(cp['data'])
        tr = datetime.strptime(e['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        tu = datetime.strptime(cd['timestamp'],'%Y-%m-%d %H:%M:%S.%f')
        ts = tu-tr
        base.append(tr.timestamp())
        x.append(ts.total_seconds())
        y.append(p['client'])

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
        fh.write(tab.to_html(full_html=False, include_plotlyjs='cdn'))
    
    return render_template('clients.html')

@app.route("/ml", methods=['POST', 'GET'])
def ml():
 
  
    c = pymongo.MongoClient()
    mc = pymongo.MongoClient('mongo',27017,username='root',password='example')
    mdb = mc["ac435faef-c2df-442e-b349-7f633d3d5523"]
    alliance = mdb["status"]

    # Assemble a dict with all validations
    validations = {}
    for post in alliance.find({'type': 'MODEL_VALIDATION'}):
        e = json.loads(post['data'])

        try:
            validations[e['modelId']].append(json.loads(e['data'])["mae"])
        except KeyError:
            validations[e['modelId']] = [json.loads(e['data'])["mae"]]
   
    rounds = []
    clients = {}

    index = 1
    count = 0
    num_clients =  4 #len(validations[list(validations.keys())[0]])

    for client in range(num_clients):
        clients['Client'+str(client)] = []

    for key in validations:

        for c in clients:
            clients[c].append(float(validations[key][count]))
            count = count + 1

        rounds.append(index)
        index = index + 1
        count = 0
    traces_data = []

    for c in clients:
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
 
  
    c = pymongo.MongoClient()
    mc = pymongo.MongoClient('mongo',27017,username='root',password='example')
    mdb = mc["ac435faef-c2df-442e-b349-7f633d3d5523"]
    alliance = mdb["status"]

    #metric = 'mae'
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

