FROM scaleoutsystems/fedn-client:develop
COPY fedn-network.yaml /app/ 
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
WORKDIR /app
