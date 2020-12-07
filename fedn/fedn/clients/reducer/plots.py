import pymongo
import json
import numpy
import plotly.graph_objs as go
from datetime import datetime,timedelta
import plotly
import os
from fedn.common.storage.db.mongo import connect_to_mongodb, drop_mongodb


class Plot:
    def __init__(self):
        try:
            self.mdb = connect_to_mongodb()
            self.alliance = self.mdb["status"]
            self.round_time = self.mdb["control.round_time"]
            self.psutil_usage = self.mdb["control.psutil_monitoring"]

        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.collection = None
            raise

    # plot metrics from DB
    def _scalar_metrics(self, metrics):
        """ Extract all scalar valued metrics from a MODEL_VALIDATON. """

        data = json.loads(metrics['data'])
        data = json.loads(data['data'])

        valid_metrics = []
        for metric, val in data.items():
            # If it can be converted to a float it is a valid, scalar metric
            try:
                val = float(val)
                valid_metrics.append(metric)
            except:
                pass

        return valid_metrics

    def create_table_plot(self):
        metrics = self.alliance.find_one({'type': 'MODEL_VALIDATION'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for mean metrics')
            table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return table

        valid_metrics = self._scalar_metrics(metrics)
        if valid_metrics == []:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No scalar metrics found')
            table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return table

        all_vals = []
        models = []
        for metric in valid_metrics:
            validations = {}
            for post in self.alliance.find({'type': 'MODEL_VALIDATION'}):
                e = json.loads(post['data'])
                try:
                    validations[e['modelId']].append(float(json.loads(e['data'])[metric]))
                except KeyError:
                    validations[e['modelId']] = [float(json.loads(e['data'])[metric])]

            vals = []
            models = []
            for model, data in validations.items():
                vals.append(numpy.mean(data))
                models.append(model)
            all_vals.append(vals)

        header_vals = valid_metrics
        models.reverse()
        values = [models]
        print(all_vals, flush=True)
        for vals in all_vals:
            vals.reverse()
            values.append(vals)

        fig = go.Figure(data=[go.Table(
            header=dict(values=['Model ID'] + header_vals,
                        line_color='darkslategray',
                        fill_color='lightskyblue',
                        align='left'),

            cells=dict(values=values,  # 2nd column
                       line_color='darkslategray',
                       fill_color='lightcyan',
                       align='left'))
        ])

        fig.update_layout(title_text='Summary: mean metrics')
        table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return table

    def create_timeline_plot(self):
        trace_data = []
        x = []
        y = []
        base = []
        for p in self.alliance.find({'type': 'MODEL_UPDATE_REQUEST'}):
            e = json.loads(p['data'])
            cid = e['correlationId']
            for cc in self.alliance.find({'sender': p['sender'], 'type': 'MODEL_UPDATE'}):
                da = json.loads(cc['data'])
                if da['correlationId'] == cid:
                    cp = cc

            cd = json.loads(cp['data'])
            tr = datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            tu = datetime.strptime(cd['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            ts = tu - tr
            base.append(tr.timestamp())
            x.append(ts.total_seconds()/60.0)
            y.append(p['sender']['name'])

        trace_data.append(go.Bar(
            x=y,
            y=x,
            marker=dict(color='royalblue'),
            name="Training",
        ))

        x = []
        y = []
        base = []
        for p in self.alliance.find({'type': 'MODEL_VALIDATION_REQUEST'}):
            e = json.loads(p['data'])
            cid = e['correlationId']
            for cc in self.alliance.find({'sender': p['sender'], 'type': 'MODEL_VALIDATION'}):
                da = json.loads(cc['data'])
                if da['correlationId'] == cid:
                    cp = cc
            cd = json.loads(cp['data'])
            tr = datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            tu = datetime.strptime(cd['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            ts = tu - tr
            base.append(tr.timestamp())
            x.append(ts.total_seconds()/60.0)
            y.append(p['sender']['name'])

        trace_data.append(go.Bar(
            x=y,
            y=x,
            marker=dict(color='lightskyblue'),
            name="Validation",
        ))

        layout = go.Layout(
            barmode='stack',
            showlegend=True,
        )

        fig = go.Figure(data=trace_data, layout=layout)
        fig.update_xaxes(title_text='Alliance/client')
        fig.update_yaxes(title_text='Time (Min)')
        fig.update_layout(title_text='Alliance timeline')
        timeline = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return timeline

    def create_ml_plot(self):
        metrics = self.alliance.find_one({'type': 'MODEL_VALIDATION'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for Mean Absolute Error')
            ml = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return ml

        data = json.loads(metrics['data'])
        data = json.loads(data['data'])
        valid_metrics = []
        for metric, val in data.items():
            # Check if scalar - is this robust ?
            if isinstance(val, float):
                valid_metrics.append(metric)

        # Assemble a dict with all validations
        validations = {}
        clients = {}

        for post in self.alliance.find({'type': 'MODEL_VALIDATION'}):
            try:
                e = json.loads(post['data'])
                clients[post['sender']['name']].append(json.loads(e['data'])[metric])
            except KeyError:
                clients[post['sender']['name']] = []

        rounds = []
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
        ml = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return ml

    def create_box_plot(self):
        metrics = self.alliance.find_one({'type': 'MODEL_VALIDATION'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for metric distribution over alliance '
                                         'participants')
            box = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return box

        valid_metrics = self._scalar_metrics(metrics)
        if valid_metrics == []:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No scalar metrics found')
            box = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return box

        # Just grab the first metric in the list.
        # TODO: Let the user choose, or plot all of them.
        if "accuracy" in valid_metrics:
            metric = "accuracy"
        else:
            metric = valid_metrics[0]
        validations = {}
        for post in self.alliance.find({'type': 'MODEL_VALIDATION'}):
            e = json.loads(post['data'])
            try:
                validations[e['modelId']].append(float(json.loads(e['data'])[metric]))
            except KeyError:
                validations[e['modelId']] = [float(json.loads(e['data'])[metric])]

        box = go.Figure()

        x = []
        y = []
        box_trace = []
        for model_id, acc in validations.items():
            x.append(model_id)
            y.append(numpy.mean([float(i) for i in acc]))
            if len(acc) >= 2:
                box.add_trace(go.Box(y=acc, name=str(model_id), marker_color="royalblue", showlegend=False))

        rounds = list(range(len(y)))
        box.add_trace(go.Scatter(
            x=x,
            y=y,
            name='Mean'
        ))

        box.update_xaxes(title_text='Model ID')
        box.update_yaxes(tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        box.update_layout(title_text='Metric distribution over alliance participants: {}'.format(metric))
        box = json.dumps(box, cls=plotly.utils.PlotlyJSONEncoder)
        return box

    def create_round_plot(self):
        metrics = self.round_time.find_one({'key': 'round_time'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for round time')
            ml = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return ml

        for post in self.round_time.find({'key': 'round_time'}):
            rounds = post['round']
            traces_data = post['round_time']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds,
            y=traces_data,
            mode='lines+markers',
            name='Time'
        ))

        fig.update_xaxes(title_text='Round')
        fig.update_yaxes(title_text='Time (s)')
        fig.update_layout(title_text='Round time')
        round_t = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return round_t

    def create_cpu_plot(self):
        metrics = self.psutil_usage.find_one({'key': 'cpu_mem_usage'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for CPU usage')
            cpu = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return cpu

        for post in self.psutil_usage.find({'key': 'cpu_mem_usage'}):
            cpu = post['cpu']
            mem = post['mem']
            ps_time = post['time']
            round = post['round']

        # Create figure with secondary y-axis
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=ps_time,
            y=cpu,
            mode='lines+markers',
            name='CPU (%)'
        ))

        fig.add_trace(go.Scatter(
            x=ps_time,
            y=mem,
            mode='lines+markers',
            name='MEM (%)'
        ))

        fig.add_trace(go.Scatter(
            x=ps_time,
            y=round,
            mode='lines+markers',
            name='Round',
        ), secondary_y=True)

        fig.update_xaxes(title_text='Date Time')
        fig.update_yaxes(title_text='Percentage (%)')
        fig.update_yaxes(title_text="Round", secondary_y=True)
        fig.update_layout(title_text='CPU loads and memory usage')
        cpu = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return cpu