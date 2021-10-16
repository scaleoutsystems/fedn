import pymongo
import json
import numpy
import plotly.graph_objs as go
from datetime import datetime, timedelta
import plotly
import os
from fedn.common.storage.db.mongo import connect_to_mongodb, drop_mongodb
import math

import plotly.express as px
import geoip2.database
import pandas as pd


from bokeh.embed import json_item
from bokeh.plotting import figure, from_networkx
from bokeh.resources import CDN
from bokeh.sampledata.iris import flowers

import networkx
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine

import pandas as pd
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Range1d, TapTool)
from bokeh.palettes import Spectral4

class Plot:
    """

    """

    def __init__(self, statestore):
        try:
            statestore_config = statestore.get_config()
            self.mdb = connect_to_mongodb(statestore_config['mongo_config'], statestore_config['network_id'])
            self.status = self.mdb['control.status']
            self.round_time = self.mdb["control.round_time"]
            self.combiner_round_time = self.mdb["control.combiner_round_time"]
            self.psutil_usage = self.mdb["control.psutil_monitoring"]
            self.network_clients = self.mdb["network.clients"]

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
        """

        :return:
        """
        metrics = self.status.find_one({'type': 'MODEL_VALIDATION'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for table mean metrics')
            table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return False

        valid_metrics = self._scalar_metrics(metrics)
        if valid_metrics == []:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No scalar metrics found')
            table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return False

        all_vals = []
        models = []
        for metric in valid_metrics:
            validations = {}
            for post in self.status.find({'type': 'MODEL_VALIDATION'}):
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
        """

        :return:
        """
        trace_data = []
        x = []
        y = []
        base = []
        for p in self.status.find({'type': 'MODEL_UPDATE_REQUEST'}):
            e = json.loads(p['data'])
            cid = e['correlationId']
            for cc in self.status.find({'sender': p['sender'], 'type': 'MODEL_UPDATE'}):
                da = json.loads(cc['data'])
                if da['correlationId'] == cid:
                    cp = cc

            cd = json.loads(cp['data'])
            tr = datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            tu = datetime.strptime(cd['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            ts = tu - tr
            base.append(tr.timestamp())
            x.append(ts.total_seconds() / 60.0)
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
        for p in self.status.find({'type': 'MODEL_VALIDATION_REQUEST'}):
            e = json.loads(p['data'])
            cid = e['correlationId']
            for cc in self.status.find({'sender': p['sender'], 'type': 'MODEL_VALIDATION'}):
                da = json.loads(cc['data'])
                if da['correlationId'] == cid:
                    cp = cc
            cd = json.loads(cp['data'])
            tr = datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            tu = datetime.strptime(cd['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
            ts = tu - tr
            base.append(tr.timestamp())
            x.append(ts.total_seconds() / 60.0)
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

    def create_client_training_distribution(self):
        """

        :return:
        """
        training = []
        for p in self.status.find({'type': 'MODEL_UPDATE'}):
            e = json.loads(p['data'])
            meta = json.loads(e['meta'])
            training.append(meta['exec_training'])

        if not training:
            return False
        fig = go.Figure(data=go.Histogram(x=training))
        fig.update_layout(title_text='Client model training time, mean: {}'.format(numpy.mean(training)))
        histogram = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return histogram

    def create_client_plot(self):
        """

        :return:
        """
        processing = []
        upload = []
        download = []
        training = []
        for p in self.status.find({'type': 'MODEL_UPDATE'}):
            e = json.loads(p['data'])
            meta = json.loads(e['meta'])
            upload.append(meta['upload_model'])
            download.append(meta['fetch_model'])
            training.append(meta['exec_training'])
            processing.append(meta['processing_time'])

        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "histogram"}]])

        fig.update_layout(
            template="simple_white",
            xaxis=dict(title_text="Seconds"),
            title="Total mean client processing time: {}".format(numpy.mean(processing)),
            showlegend=True
        )
        if not processing:
            return False
        data = [numpy.mean(training), numpy.mean(upload), numpy.mean(download)]
        labels = ["Training", "Model upload", "Model download"]
        fig.add_trace(go.Pie(labels=labels, values=data), row=1, col=1)

        fig.add_trace(go.Histogram(x=training), row=1, col=2)

        client_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return client_plot

    def create_combiner_plot(self):
        """

        :return:
        """
        waiting = []
        aggregation = []
        model_load = []
        combination = []
        for round in self.mdb['control.round'].find():
            try:
                for combiner in round['combiners']:
                    data = combiner
                    stats = data['local_round']['1']
                    ml = stats['aggregation_time']['time_model_load']
                    ag = stats['aggregation_time']['time_model_aggregation']
                    combination.append(stats['time_combination'])
                    waiting.append(stats['time_combination'] - ml - ag)
                    model_load.append(ml)
                    aggregation.append(ag)
            except:
                pass

        labels = ['Waiting for updates', 'Aggregating model updates', 'Loading model updates']
        val = [numpy.mean(waiting), numpy.mean(aggregation), numpy.mean(model_load)]
        fig = go.Figure()

        fig.update_layout(
            template="simple_white",
            title="Total mean combiner round time: {}".format(numpy.mean(combination)),
            showlegend=True
        )
        if not combination:
            return False
        fig.add_trace(go.Pie(labels=labels, values=val))
        combiner_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return combiner_plot

    def fetch_valid_metrics(self):
        """

        :return:
        """
        metrics = self.status.find_one({'type': 'MODEL_VALIDATION'})
        valid_metrics = self._scalar_metrics(metrics)
        return valid_metrics

    def create_box_plot(self, metric):
        """

        :param metric:
        :return:
        """
        metrics = self.status.find_one({'type': 'MODEL_VALIDATION'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for metric distribution over  '
                                         'participants')
            box = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return box

        valid_metrics = self._scalar_metrics(metrics)
        if valid_metrics == []:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No scalar metrics found')
            box = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return box

        validations = {}
        for post in self.status.find({'type': 'MODEL_VALIDATION'}):
            e = json.loads(post['data'])
            try:
                validations[e['modelId']].append(float(json.loads(e['data'])[metric]))
            except KeyError:
                validations[e['modelId']] = [float(json.loads(e['data'])[metric])]

        # Make sure validations are plotted in chronological order
        model_trail = self.mdb.control.model.find_one({'key': 'model_trail'})
        model_trail_ids = model_trail['model']
        validations_sorted = []
        for model_id in model_trail_ids:
            try:
                validations_sorted.append(validations[model_id])
            except:
                pass

        validations = validations_sorted

        box = go.Figure()

        x = []
        y = []
        box_trace = []
        for j, acc in enumerate(validations):
            # x.append(j)
            y.append(numpy.mean([float(i) for i in acc]))
            if len(acc) >= 2:
                box.add_trace(go.Box(y=acc, name=str(j), marker_color="royalblue", showlegend=False,
                                     boxpoints=False))
            else:
                box.add_trace(go.Scatter(x=[str(j)], y=[y[j]], showlegend=False))

        rounds = list(range(len(y)))
        box.add_trace(go.Scatter(
            x=rounds,
            y=y,
            name='Mean'
        ))

        box.update_xaxes(title_text='Rounds')
        box.update_yaxes(tickvals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        box.update_layout(title_text='Metric distribution over clients: {}'.format(metric),
                          margin=dict(l=20, r=20, t=45, b=20))
        box = json.dumps(box, cls=plotly.utils.PlotlyJSONEncoder)
        return box

    def create_round_plot(self):
        """

        :return:
        """
        trace_data = []
        metrics = self.round_time.find_one({'key': 'round_time'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for round time')
            ml = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return False

        for post in self.round_time.find({'key': 'round_time'}):
            rounds = post['round']
            traces_data = post['round_time']

        trace_data.append(go.Scatter(
            x=rounds,
            y=traces_data,
            mode='lines+markers',
            name='Reducer'
        ))

        for rec in self.combiner_round_time.find({'key': 'combiner_round_time'}):
            c_traces_data = rec['round_time']

        trace_data.append(go.Scatter(
            x=rounds,
            y=c_traces_data,
            mode='lines+markers',
            name='Combiner'
        ))

        fig = go.Figure(data=trace_data)
        fig.update_xaxes(title_text='Round')
        fig.update_yaxes(title_text='Time (s)')
        fig.update_layout(title_text='Round time')
        round_t = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return round_t

    def create_cpu_plot(self):
        """

        :return:
        """
        metrics = self.psutil_usage.find_one({'key': 'cpu_mem_usage'})
        if metrics == None:
            fig = go.Figure(data=[])
            fig.update_layout(title_text='No data currently available for MEM and CPU usage')
            cpu = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return False

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

    def get_client_df(self):
        clients = self.network_clients
        df = pd.DataFrame(list(clients.find()))
        return df

    def make_netgraph_plot(self, df):
        G = networkx.from_pandas_edgelist(df, 'source', 'target')
        degrees = dict(networkx.degree(G))
        density = networkx.density(G)

        # membership = list(df.role)
        node_list = list(G.nodes())
        edge_list = G.edges()

        # count_dict = {k: v for k, v in zip(node_list, membership)}
        # print('count_dict', count_dict)

        adjusted_node_size = dict([(node, degree + 3) for node, degree in networkx.degree(G)])

        networkx.set_node_attributes(G, name='degree', values=degrees)
        networkx.set_node_attributes(G, name='density', values=density)
        # networkx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
        # networkx.set_node_attributes(G, name='count_dict', values=count_dict)
        # networkx.set_node_attributes(G, name='role', values=count_dict)

        # https://embed.plnkr.co/plunk/haezsh
        # Choose a title!
        title = 'FEDn Network'

        # Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [("Name", "@index"), ("Degree", "@degree"),
                          #                   ("count_dict", "@count_dict"),
                          ]

        # Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips=None,
                      tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                      x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=None)

        plot.xgrid.grid_line_color = None
        plot.ygrid.grid_line_color = None
        plot.axis.visible = False
        plot.yaxis.visible = False

        plot.add_tools(HoverTool(tooltips=HOVER_TOOLTIPS), TapTool(), BoxSelectTool())
        # Create a network graph object with spring layout
        # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
        network_graph = from_networkx(G, networkx.spring_layout, scale=10, center=(0, 0))

        # Set node size and color
        network_graph.node_renderer.glyph = Circle(size=15, fill_color='skyblue')

        # Set edge opacity and width
        # network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

        # graph_renderer = from_networkx(G, networkx.spring_layout, scale=1, center=(0,0))

        # network_graph.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
        network_graph.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        network_graph.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

        network_graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
        network_graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        network_graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        network_graph.selection_policy = NodesAndLinkedEdges()
        # network_graph.inspection_policy = EdgesAndLinkedNodes()

        # Add network graph to the plot
        plot.renderers.append(network_graph)

        # #Add Labels
        # x, y = zip(*network_graph.layout_provider.graph_layout.values())
        # node_labels = list(G.nodes())
        # source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
        # labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color=None, text_font_size='10px', background_fill_alpha=.7)
        # plot.renderers.append(labels)

        # show(plot)
        # save(plot, filename=f"{title}.html")
        return plot