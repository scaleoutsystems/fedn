FROM python:3.8.5
RUN pip install numpy==1.18.5 h5py==2.10.0 keras pandas tensorflow sklearn
RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
RUN pip install -e /app/fedn
WORKDIR /app
