FROM rust:1.87-slim

WORKDIR /workspace
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN cargo test --package doredore-core
