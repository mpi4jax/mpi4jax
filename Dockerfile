FROM       nvidia/cuda:11.2.2-cudnn8-devel

ENV        DEBIAN_FRONTEND="noninteractive"

RUN        apt-get update -y && \
           apt-get install -y --no-install-recommends \
           openmpi-bin libopenmpi-dev \
           git-all \
           python3-dev \
           python3-pip \
           python3-wheel \
           python3-setuptools && \
           rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN        ln -s /usr/bin/python3 /usr/bin/python
RUN        ln -s /usr/bin/pip3 /usr/bin/pip

RUN        useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER       docker
RUN        mkdir /home/docker/workdir

RUN        pip3 install pytest
RUN        pip3 install --no-cache-dir -U install setuptools pip

ENV        SHELL=/bin/bash

WORKDIR    /home/docker/workdir

