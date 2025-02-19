ARG OS_VERSION=24.04
ARG LIB=uhd
ARG LIB_VERSION=4.7.0.0
ARG MARCH=native
ARG NUM_CORES=""

FROM ubuntu:$OS_VERSION

ENV CONFIG="configs/basic_jammer.conf"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    cmake \
    make \
    gcc \
    g++ \
    pkg-config \
    uhd-host \
    libboost-all-dev \
    libuhd-dev

RUN mkdir -p /app

WORKDIR /app

COPY CMakeLists.txt .
COPY hdr hdr
COPY src src
COPY cmake cmake

RUN mkdir -p /app/build

WORKDIR /app/build

RUN cmake ../ && \
    make -j$(nproc) && \
    make install

WORKDIR /app

COPY configs configs

CMD [ "sh", "-c", "/usr/local/bin/rtue \"${CONFIG}\" $ARGS" ]
