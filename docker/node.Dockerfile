FROM rust:1.87-slim

WORKDIR /workspace
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN cd doredore-js && npm install && npm run build && npm pack

RUN mkdir -p /tmp/doredore-node-test \
    && cd /tmp/doredore-node-test \
    && npm init -y \
    && npm install /workspace/doredore-js/doredore-0.1.0.tgz \
    && cp /workspace/examples/nodejs/basic.js /tmp/doredore-node-test/basic.js \
    && node /tmp/doredore-node-test/basic.js
