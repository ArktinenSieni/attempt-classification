# Our base image
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install lower level dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y curl python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# pip requirements
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir scikit-learn thundersvm pandas