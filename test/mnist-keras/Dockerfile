FROM python:3.8.9
RUN pip install -e git://github.com/scaleoutsystems/fedn.git@master#egg=fedn\&subdirectory=fedn
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
COPY fedn-network.yaml /app/ 


