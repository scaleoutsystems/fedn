@app.route('/metric_type', methods=['GET', 'POST'])
def change_features():
    """

    :return:
    """
    feature = request.args['selected']
    plot = Plot(self.control.statestore)
    graphJSON = plot.create_box_plot(feature)
    return graphJSON