
@app.route('/dashboard')
def dashboard():
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
    combiners_plot = plot.create_combiner_plot()

    timeline_plot = None
    table_plot = None
    clients_plot = plot.create_client_plot()
    client_histogram_plot = plot.create_client_histogram_plot()

    return render_template('dashboard.html', show_plot=True,
                           table_plot=table_plot,
                           timeline_plot=timeline_plot,
                           clients_plot=clients_plot,
                           client_histogram_plot=client_histogram_plot,
                           combiners_plot=combiners_plot,
                           configured=True
                           )