FROM ghcr.io/oran-testing/components_base AS builder
WORKDIR /jammer

COPY . .

RUN mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install

    
FROM alpine:latest
ENV PYTHONUNBUFFERED=1
RUN apk add --no-cache libstdc++ ca-certificates && update-ca-certificates || true

COPY --from=builder /usr/local /usr/local

ENV ARGS=""
CMD ["sh", "-c", "/usr/local/bin/jammer --config /jammer.yaml $ARGS"]
