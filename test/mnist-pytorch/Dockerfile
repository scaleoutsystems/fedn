FROM pytorch/pytorch
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn
COPY fedn-network.yaml /app/
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt