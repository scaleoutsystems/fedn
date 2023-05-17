@app.route('/networkgraph')
        def network_graph():

            try:
                plot = Plot(self.control.statestore)
                result = netgraph()
                df_nodes = pd.DataFrame(result['nodes'])
                df_edges = pd.DataFrame(result['edges'])
                graph = plot.make_netgraph_plot(df_edges, df_nodes)
                return json.dumps(json_item(graph, "myplot"))
            except Exception:
                raise
                # return ''


@app.route('/network')
        def network():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

            not_configured = self.check_configured()
            if not_configured:
                return not_configured
            plot = Plot(self.control.statestore)
            round_time_plot = plot.create_round_plot()
            mem_cpu_plot = plot.create_cpu_plot()
            combiner_info = combiner_status()
            active_clients = client_status()
            return render_template('network.html', network_plot=True,
                                   round_time_plot=round_time_plot,
                                   mem_cpu_plot=mem_cpu_plot,
                                   combiner_info=combiner_info,
                                   active_clients=active_clients['active_clients'],
                                   active_trainers=active_clients['active_trainers'],
                                   active_validators=active_clients['active_validators'],
                                   configured=True
                                   )