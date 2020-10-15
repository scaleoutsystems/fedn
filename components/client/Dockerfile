FROM morganekmefjord/basesci:latest

RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
RUN pip install -e /app/fedn
WORKDIR /app

# Clone the dashboard files into the docker container and install package drom dev branch
RUN git clone --branch development https://github.com/flask-dashboard/Flask-MonitoringDashboard.git
RUN pip install -e /app/Flask-MonitoringDashboard