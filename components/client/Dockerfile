FROM morganekmefjord/basesci:latest
RUN adduser --disabled-password --gecos '' fednuser \
    && adduser fednuser sudo \
    && echo '%sudo ALL=(ALL:ALL) ALL' >> /etc/sudoers


RUN mkdir /app
RUN mkdir /app/certs
COPY fedn /app/fedn
RUN pip install -e /app/fedn
RUN chown -R fednuser /app
USER fednuser
WORKDIR /app
