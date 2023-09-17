from nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TZ=America/Argentina/Buenos_Aires
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY requirements.txt .
RUN apt-get --fix-missing update && apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    sox
RUN pip3 install -r requirements.txt

