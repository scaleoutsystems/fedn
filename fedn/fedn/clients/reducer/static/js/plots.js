$('#plot_type').on('change',function(){

    $.ajax({
        url: "/plot_type",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        data: {
            'selected': document.getElementById('plot_type').value

        },
        dataType:"json",
        success: function (data) {
            Plotly.newPlot('bargraph', data );
        }
    });
})
