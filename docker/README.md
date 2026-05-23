Docker-based verification environments for distribution checks.

Build and run each image from the repo root:

Rust core:
  docker build -f docker/rust.Dockerfile -t doredore-rust-check .

Python wheel:
  docker build -f docker/python.Dockerfile -t doredore-python-check .

Node.js (napi-rs):
  docker build -f docker/node.Dockerfile -t doredore-node-check .

Ruby (FFI):
  docker build -f docker/ruby.Dockerfile -t doredore-ruby-check .
