from numpy.core.einsumfunc import _flop_count
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

import networkx
import pandas as pd
from bokeh.models import (Circle, Label, LabelSet,
                          MultiLine, NodesAndLinkedEdges, Range1d, ColumnDataSource)
from bokeh.plotting import figure, from_networkx
from bokeh.palettes import Spectral8

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
        active_clients = df['status']=="active"
        print(df[active_clients])
        return df
    
    def make_single_node_plot(self):
        """
        Plot single node graph with reducer

        :return: Bokeh plot with the graph
        :rtype: bokeh.plotting.figure.Figure
        """
        HOVER_TOOLTIPS = [
            ("Name", "@name"),
            ("Role", "@role"),
            ("Status", "@status"),
            ("Id", "@index"),
            ]
        
        G = networkx.Graph()
        G.add_node("reducer", adjusted_node_size=20, role='reducer',
                    status='active', 
                    name='reducer',
                    color_by_this_attribute=Spectral8[0])
        network_graph = from_networkx(G, networkx.spring_layout)
        network_graph.node_renderer.glyph = Circle(size=20, fill_color = Spectral8[0])
        network_graph.node_renderer.hover_glyph = Circle(size=20, fill_color='white',
                                                         line_width=2)
        network_graph.node_renderer.selection_glyph = Circle(size=20,
                                                             fill_color='white', line_width=2)
        plot = figure(tooltips=HOVER_TOOLTIPS, tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                    width=725, height=460, sizing_mode='stretch_width',
                    x_range=Range1d(-1.5, 1.5), y_range=Range1d(-1.5, 1.5))
        
        plot.renderers.append(network_graph)

        plot.axis.visible = False
        plot.grid.visible = False
        plot.outline_line_color = None

        label = Label(x=0, y=0, text='reducer',
                     background_fill_color='#4bbf73', text_font_size='15px',
                     background_fill_alpha=.7, x_offset=-20, y_offset=10)
        
        plot.add_layout(label)
        return plot
        

        

    def make_netgraph_plot(self, df, df_nodes):
        """
        Create FEDn network visualization.

        :param df: pandas dataframe with defined edges
        :type df: pandas.Dataframe
        :param df_nodes:pandas dataframe with defined nodes
        :type df_nodes: pandas.Dataframe
        :return: Bokeh plot with the graph
        :rtype: bokeh.plotting.figure.Figure
        """

        if df.empty:
            #no combiners and thus no clients, plot only reducer
            plot = self.make_single_node_plot()
            return plot
           
        G = networkx.from_pandas_edgelist(df, 'source', 'target', create_using=networkx.Graph())
        degrees = dict(networkx.degree(G))
        networkx.set_node_attributes(G, name='degree', values=degrees)

        number_to_adjust_by = 20
        adjusted_node_size = dict([(node, degree + number_to_adjust_by) for node, degree in networkx.degree(G)])
        networkx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
        
        # community
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G)
        # Create empty dictionaries
        modularity_class = {}
        modularity_color = {}
        # Loop through each community in the network
        for community_number, community in enumerate(communities):
            # For each member of the community, add their community number and a distinct color
            for name in community:
                modularity_class[name] = community_number
                modularity_color[name] = Spectral8[community_number]

        # Add modularity class and color as attributes from the network above
        networkx.set_node_attributes(G, modularity_class, 'modularity_class')
        networkx.set_node_attributes(G, modularity_color, 'modularity_color')

        node_role = {k:v for k,v in zip(df_nodes.id, df_nodes.role)}
        networkx.set_node_attributes(G, node_role, 'role')
        
        node_status = {k:v for k,v in zip(df_nodes.id, df_nodes.status)}
        networkx.set_node_attributes(G, node_status, 'status')

        node_name = {k:v for k,v in zip(df_nodes.id, df_nodes.name)}
        networkx.set_node_attributes(G, node_name, 'name')

        
        # Choose colors for node and edge highlighting
        node_highlight_color = 'white'
        edge_highlight_color = 'black'

        # Choose attributes from G network to size and color by — setting manual
        # size (e.g. 10) or color (e.g. 'skyblue') also allowed
        size_by_this_attribute = 'adjusted_node_size'
        color_by_this_attribute = 'modularity_color'

        # Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Name", "@name"),
            ("Role", "@role"),
            ("Status", "@status"),
            ("Id", "@index"),
        ]
        # Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips=HOVER_TOOLTIPS,
                      tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                      width=725, height=460, sizing_mode='stretch_width',
                      x_range=Range1d(-1.5, 1.5), y_range=Range1d(-1.5, 1.5))

        # Create a network graph object
        # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
        # if one like lock reducer add args: pos={'reducer':(0,1)}, fixed=['reducer']
        network_graph = from_networkx(G, networkx.spring_layout, scale=1, center=(0, 0), seed=45)

        # Set node sizes and colors according to node degree (color as category from attribute)
        network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)
        # Set node highlight colors
        network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color,
                                                         line_width=2)
        network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute,
                                                             fill_color=node_highlight_color, line_width=2)
        
        # Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
        # Set edge highlight colors
        network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
        network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)

        # Highlight nodes and edges
        network_graph.selection_policy = NodesAndLinkedEdges()
        network_graph.inspection_policy = NodesAndLinkedEdges()

        plot.renderers.append(network_graph)
        
        #Node labels, red if status is offline, green is active
        x, y = zip(*network_graph.layout_provider.graph_layout.values())
        node_names = list(G.nodes(data='name'))
        node_status = list(G.nodes(data='status'))

        idx_offline = []
        idx_online = []
        node_labels = []
        for e, n in enumerate(node_names):
            if node_status[e][1] == 'active':
                idx_online.append(e)
            else:
                idx_offline.append(e)
            node_labels.append(n[1])

        source_on = ColumnDataSource({'x': numpy.asarray(x)[idx_online], 'y': numpy.asarray(y)[idx_online], 'name': numpy.asarray(node_labels)[idx_online]})
        labels = LabelSet(x='x', y='y', text='name', source=source_on, background_fill_color='#4bbf73', text_font_size='15px',
                          background_fill_alpha=.7, x_offset=-20, y_offset=10)

        plot.renderers.append(labels)

        
        source_off = ColumnDataSource({'x': numpy.asarray(x)[idx_offline], 'y': numpy.asarray(y)[idx_offline], 'name': numpy.asarray(node_labels)[idx_offline]})
        labels = LabelSet(x='x', y='y', text='name', source=source_off, background_fill_color='#d9534f', text_font_size='15px',
                          background_fill_alpha=.7, x_offset=-20, y_offset=10)

        plot.renderers.append(labels)

        plot.axis.visible = False
        plot.grid.visible = False
        plot.outline_line_color = None
        return plot