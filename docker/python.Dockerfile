FROM rust:1.87-slim

WORKDIR /workspace
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir maturin
RUN cd doredore-py && maturin build --release
RUN pip install --no-cache-dir doredore-py/target/wheels/doredore-0.1.0-*.whl

RUN python examples/python/basic.py
