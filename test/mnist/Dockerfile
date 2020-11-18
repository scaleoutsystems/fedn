FROM morganekmefjord/basesci:latest
RUN mkdir -p /app/client
RUN mkdir /app/certs
#COPY fedn /app/fedn
#RUN pip install -e /app/fedn
COPY fedn-network.yaml /app/ 
WORKDIR /app
RUN python -m pip install -e "git+https://github.com/scaleoutsystems/fedn.git@develop#egg=fedn&subdirectory=fedn"
