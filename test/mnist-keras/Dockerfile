FROM python:3.8.5
RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
COPY fedn-network.yaml /app/ 


