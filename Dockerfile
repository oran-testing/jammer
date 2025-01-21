ARG OS_VERSION=24.04
ARG LIB=uhd
ARG LIB_VERSION=4.7.0.0
ARG MARCH=native
ARG NUM_CORES=""

FROM ubuntu:$OS_VERSION

ENV CONFIG="configs/basic_jammer.yaml"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        gnuradio \
        gr-osmosdr



WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install -r  requirements.txt --break-system-packages

COPY jammer.py .
COPY ./configs configs

CMD ["python3", "jammer.py", "$CONFIG"]

