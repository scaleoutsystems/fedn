$('#metric_type').on('change', function () {
    $.ajax({
        url: "/metric_type",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        data: {
            'selected': document.getElementById('metric_type').value

        },
        dataType: "json",
        success: function (data) {
            Plotly.newPlot('box', data);
        }
    });
})