FROM       nvidia/cuda:11.2.2-devel-ubuntu20.04
FROM	   python

RUN 	   apt-get update
RUN 	   apt-get install -y openmpi-bin libopenmpi-dev
RUN        pip install -U pip
RUN        pip install pytest


ENV        SHELL=/bin/bash

COPY       . /testing

WORKDIR    /testing

