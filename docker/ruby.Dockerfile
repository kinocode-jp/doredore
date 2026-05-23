FROM rust:1.87-slim

WORKDIR /workspace
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    ruby \
    ruby-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN cargo build --package doredore-rb --release
RUN cd doredore-rb && gem build doredore.gemspec

ENV RUBYLIB="/workspace/doredore-rb/lib"
RUN ruby examples/ruby/basic.rb
