FROM scaleoutsystems/fedn-client:develop
COPY test/pytorch_mnist/fedn-network.yaml /app/ 
COPY test/pytorch_mnist/requirements.txt /app/requirements.txt
COPY fedn /app/fedn
RUN pip install -r /app/requirements.txt &&\
    pip install -e /app/fedn
WORKDIR /app
