FROM jupyter/minimal-notebook:latest
COPY . /home/jovyan/stackn
USER root
RUN /bin/bash -c "pip install /home/jovyan/stackn/"
USER $NB_UID
