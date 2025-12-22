FROM rust:1.90-slim-bullseye

# 1. Install dependencies required for Rust and mdbook
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install mdbook and plugins
RUN cargo install mdbook mdbook-mermaid

WORKDIR /workspaces/robotics-handbook
CMD ["/bin/bash"]