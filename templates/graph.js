var ctx = document.getElementById('myChart').getContext('2d');
var myChart;

function createChart(chartType) {
    if (myChart) {
        myChart.destroy();
    }

    myChart = new Chart(ctx, {
        type: chartType,
        data: {
            labels: [],
            datasets: [{
                label: 'Random Data',
                data: [],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function generateData() {
    var numPoints = document.getElementById('numPoints').value;
    var chartType = document.getElementById('chartType').value;
    createChart(chartType);

    fetch('/random_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: 'num_points=' + numPoints
    })
        .then(response => response.json())
        .then(data => {
            myChart.data.labels = Array.from({length: data.length}, (_, i) => `Point ${i + 1}`);
            myChart.data.datasets[0].data = data;
            myChart.update();
        });
}

document.getElementById('generateData').addEventListener('click', generateData);
createChart('line'); // Initialize with line chart