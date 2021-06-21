FROM python:3.8.9
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn